//
//  ModelDownloadView.swift
//  OLMoE.swift
//
//  Created by Luca Soldaini on 2024-09-19.
//


import SwiftUI
import Combine
import Network

func formatSize(_ size: Int64) -> String {
    let sizeInGB = Double(size) / 1_000_000_000.0
    return String(format: "%.2f GB", sizeInGB)
}

class BackgroundDownloadManager: NSObject, ObservableObject, URLSessionDownloadDelegate {
    static let shared = BackgroundDownloadManager()

    @Published var downloadProgress: Float = 0
    @Published var isDownloading = false
    @Published var downloadError: String?
    @Published var isModelReady = false
    @Published var downloadedSize: Int64 = 0
    @Published var totalSize: Int64 = 0
    @Published var currentModelID: String?

    private var networkMonitor: NWPathMonitor?
    private var backgroundSession: URLSession!
    private var downloadTask: URLSessionDownloadTask?
    private var lastUpdateTime: Date = Date()
    private var hasCheckedDiskSpace = false
    private let updateInterval: TimeInterval = 0.5 // Update UI every 0.5 seconds
    private var lastDispatchedBytesWritten: Int64 = 0
    private var artifacts: [(url: URL, destination: URL)] = []
    private var currentArtifactIndex = 0

    private override init() {
        super.init()
        #if targetEnvironment(macCatalyst)
        // Use regular session for Mac Catalyst
        backgroundSession = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        #else
        // Use background session for iOS
        let config = URLSessionConfiguration.background(withIdentifier: "ai.olmo.OLMoE.backgroundDownload")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        backgroundSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        #endif

        startNetworkMonitoring()
    }

    private func startNetworkMonitoring() {
        networkMonitor = NWPathMonitor()
        networkMonitor?.pathUpdateHandler = { path in
            DispatchQueue.main.async {
                if path.status == .unsatisfied {
                    self.downloadError = "Connection lost. Please check your internet connection."
                    self.isDownloading = false
                    self.hasCheckedDiskSpace = false
                    self.isModelReady = false
                    self.lastDispatchedBytesWritten = 0
                    self.currentModelID = nil
                    self.downloadTask?.cancel()
                }
            }
        }

        let queue = DispatchQueue(label: "NetworkMonitor")
        networkMonitor?.start(queue: queue)
    }

    /// Starts the download process.
    func startDownload(for model: ModelSpec) {
        if networkMonitor?.currentPath.status == .unsatisfied {
            return
        }

        let artifacts = buildArtifacts(for: model)
        guard !artifacts.isEmpty else { return }

        isDownloading = true
        downloadError = nil
        downloadedSize = 0
        totalSize = 0
        self.lastDispatchedBytesWritten = 0
        self.currentModelID = model.id
        self.artifacts = artifacts
        self.currentArtifactIndex = 0
        startNextArtifactDownload()
    }

    private func buildArtifacts(for model: ModelSpec) -> [(url: URL, destination: URL)] {
        var output: [(url: URL, destination: URL)] = []
        let directory = model.localDirectory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        output.append((model.downloadURL, model.localModelURL))
        if model.isMultimodal, let projectorURL = model.projectorURL, let projectorPath = model.localProjectorURL {
            output.append((projectorURL, projectorPath))
        }
        return output
    }

    private func startNextArtifactDownload() {
        guard currentArtifactIndex < artifacts.count else {
            DispatchQueue.main.async {
                self.isDownloading = false
                self.isModelReady = true
                self.currentModelID = nil
            }
            return
        }
        let artifact = artifacts[currentArtifactIndex]
        downloadTask = backgroundSession.downloadTask(with: artifact.url)
        downloadTask?.resume()
    }

    /// Handles the completion of the download task.
    /// - Parameters:
    ///   - session: The URL session that completed the task.
    ///   - downloadTask: The download task that completed.
    ///   - location: The temporary location of the downloaded file.
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard currentArtifactIndex < artifacts.count else { return }
        let destination = artifacts[currentArtifactIndex].destination

        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
            DispatchQueue.main.async {
                self.currentArtifactIndex += 1
                self.startNextArtifactDownload()
            }
        } catch {
            DispatchQueue.main.async {
                self.downloadError = "Failed to save file: \(error.localizedDescription)"
                self.isDownloading = false
            }
        }
    }

    /// Handles errors that occur during the download task.
    /// - Parameters:
    ///   - session: The URL session that completed the task.
    ///   - task: The task that completed.
    ///   - error: The error that occurred, if any.
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        DispatchQueue.main.async {
            if let error = error {
                if self.downloadError == nil {
                    self.downloadError = "Download failed: \(error.localizedDescription)"
                }
                self.isDownloading = false
                self.currentModelID = nil
                self.hasCheckedDiskSpace = false
            }
        }
    }

    /// Updates the download progress and checks for disk space during the download.
    /// - Parameters:
    ///   - session: The URL session managing the download.
    ///   - downloadTask: The download task that is writing data.
    ///   - bytesWritten: The number of bytes written in this update.
    ///   - totalBytesWritten: The total number of bytes written so far.
    ///   - totalBytesExpectedToWrite: The total number of bytes expected to be written.
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        if !hasCheckedDiskSpace {
            hasCheckedDiskSpace = true
            if !hasEnoughDiskSpace(requiredSpace: totalBytesExpectedToWrite) {
                DispatchQueue.main.async {
                    self.downloadError = "Not enough disk space available.\nNeed \(formatSize(totalBytesExpectedToWrite)) free."
                }
                downloadTask.cancel()
                return
            }
        }

        let currentTime = Date()
        if currentTime.timeIntervalSince(lastUpdateTime) >= updateInterval {
            DispatchQueue.main.async {
                // Due to async nature, older updates might run later; update progress only if data is more recent.
                guard totalBytesWritten > self.lastDispatchedBytesWritten else { return }
                self.lastDispatchedBytesWritten = totalBytesWritten

                let perFileProgress = Float(totalBytesWritten) / Float(totalBytesExpectedToWrite)
                let overallProgress = (Float(self.currentArtifactIndex) + perFileProgress) / Float(max(self.artifacts.count, 1))
                self.downloadProgress = overallProgress
                self.downloadedSize = totalBytesWritten
                self.totalSize = totalBytesExpectedToWrite
                self.lastUpdateTime = currentTime
            }
        }
    }

    /// Deletes the downloaded model file, marking it as not ready.
    func flushModel(_ model: ModelSpec) {
        do {
            if FileManager.default.fileExists(atPath: model.localDirectory.path) {
                try FileManager.default.removeItem(at: model.localDirectory)
            }
            isModelReady = false
        } catch {
            downloadError = "Failed to flush model: \(error.localizedDescription)"
        }
    }

    func cancelDownload() {
        downloadTask?.cancel()
        isDownloading = false
        currentArtifactIndex = 0
        artifacts = []
        currentModelID = nil
    }

    /// Checks if there is enough disk space available for the required space.
    /// - Parameter requiredSpace: The amount of space required in bytes.
    /// - Returns: A boolean indicating whether there is enough disk space.
    private func hasEnoughDiskSpace(requiredSpace: Int64) -> Bool {
        let fileURL = URL(fileURLWithPath: NSHomeDirectory())
        do {
            let values = try fileURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            if let availableCapacity = values.volumeAvailableCapacityForImportantUsage {
                return availableCapacity > requiredSpace
            }
        } catch {
            print("Error retrieving available disk space: \(error.localizedDescription)")
        }
        return false
    }
}

/// A view that displays the model download progress and status.
struct ModelDownloadView: View {
    @StateObject private var downloadManager = BackgroundDownloadManager.shared
    @StateObject private var modelStore = ModelStore.shared
    var onSelectModel: ((ModelSpec) -> Void)? = nil
    @State private var showCustomModelSheet = false
    @State private var selectionError: String?

    public var body: some View {
        ZStack {
            Color("BackgroundColor")
                .edgesIgnoringSafeArea(.all)

            ScrollView {
                VStack(spacing: 24) {
                    Text("Models")
                        .font(.telegraf(.medium, size: 32))
                        .foregroundColor(Color("TextColor"))

                    Button("Add Custom Model") {
                        showCustomModelSheet = true
                    }
                    .buttonStyle(.PrimaryButton)

                    ForEach(modelStore.models) { model in
                        ModelRow(
                            model: model,
                            isCustom: modelStore.isCustom(model),
                            isSelected: modelStore.selectedModelID == model.id,
                            isReady: modelStore.isReady(model),
                            isDownloading: downloadManager.isDownloading && downloadManager.currentModelID == model.id,
                            progress: downloadManager.downloadProgress,
                            errorText: downloadManager.downloadError,
                            onSelect: {
                                modelStore.selectModel(model)
                                if modelStore.isReady(model) {
                                    onSelectModel?(model)
                                } else {
                                    selectionError = "Please download the model before selecting it."
                                }
                            },
                            onDownload: { downloadManager.startDownload(for: model) },
                            onDelete: {
                                downloadManager.flushModel(model)
                                if modelStore.isCustom(model) {
                                    modelStore.removeCustomModel(model)
                                }
                            },
                            onCancel: { downloadManager.cancelDownload() }
                        )
                    }

                    if let error = downloadManager.downloadError, downloadManager.currentModelID == nil {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                }
                .padding()
            }
            .sheet(isPresented: $showCustomModelSheet) {
                SheetWrapper {
                    CustomModelSheet(isPresented: $showCustomModelSheet, modelStore: modelStore)
                }
            }
            .alert("Model Not Ready", isPresented: Binding(get: {
                selectionError != nil
            }, set: { newValue in
                if !newValue { selectionError = nil }
            })) {
                Button("OK", role: .cancel) {
                    selectionError = nil
                }
            } message: {
                Text(selectionError ?? "")
            }

            Ai2LogoView(applyMacCatalystPadding: true)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        }
    }
}

private struct ModelRow: View {
    let model: ModelSpec
    let isCustom: Bool
    let isSelected: Bool
    let isReady: Bool
    let isDownloading: Bool
    let progress: Float
    let errorText: String?
    let onSelect: () -> Void
    let onDownload: () -> Void
    let onDelete: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(model.displayName)
                    .font(.headline)
                    .foregroundColor(Color("TextColor"))
                if model.isMultimodal {
                    Text("Vision")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color("Surface"))
                        .cornerRadius(12)
                }
                Spacer()
                if isSelected {
                    Text("Selected")
                        .font(.caption)
                        .foregroundColor(Color("LightGreen"))
                }
            }

            Text(isReady ? (model.isMultimodal ? "Model + projector ready" : "Model ready") : (model.isMultimodal ? "Requires model + projector" : "Not downloaded"))
                .font(.subheadline)
                .foregroundColor(Color("TextColor").opacity(0.7))

            if isDownloading {
                ProgressView(value: progress, total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle())
            }

            HStack {
                Button("Select", action: onSelect)
                    .buttonStyle(.PrimaryButton)
                    .disabled(!isReady)

                Button("Download", action: onDownload)
                    .buttonStyle(.SecondaryButton)
                    .disabled(isDownloading)

                if isDownloading {
                    Button("Cancel", action: onCancel)
                        .buttonStyle(.SecondaryButton)
                } else if isReady || isCustom {
                    Button(action: onDelete) {
                        Image(systemName: "trash")
                            .font(.caption)
                            .foregroundColor(Color("TextColor"))
                            .padding(10)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color("Surface"))
                            )
                    }
                }
            }

            if let errorText, isDownloading {
                Text(errorText)
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
        .padding()
        .background(Color("Surface"))
        .cornerRadius(20)
    }
}

private struct CustomModelSheet: View {
    enum ModelKind: String, CaseIterable, Identifiable {
        case text = "Text"
        case vision = "Multimodal"

        var id: String { rawValue }
    }

    @Binding var isPresented: Bool
    @ObservedObject var modelStore: ModelStore
    @State private var name = ""
    @State private var ggufURL = ""
    @State private var projectorURL = ""
    @State private var modelKind: ModelKind = .text
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Add Custom Model")
                .font(.telegraf(.medium, size: 24))
                .foregroundColor(Color("TextColor"))

            TextField("Model name", text: $name)
                .textFieldStyle(.roundedBorder)

            TextField("GGUF URL", text: $ggufURL)
                .textFieldStyle(.roundedBorder)

            Picker("Type", selection: $modelKind) {
                ForEach(ModelKind.allCases) { kind in
                    Text(kind.rawValue).tag(kind)
                }
            }
            .pickerStyle(.segmented)

            if modelKind == .vision {
                TextField("MMProj URL", text: $projectorURL)
                    .textFieldStyle(.roundedBorder)
            }

            if let errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .font(.caption)
            }

            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .buttonStyle(.SecondaryButton)

                Spacer()

                Button("Save") {
                    saveModel()
                }
                .buttonStyle(.PrimaryButton)
            }
        }
        .padding()
    }

    private func saveModel() {
        errorMessage = nil
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedGGUF = ggufURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedMMProj = projectorURL.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedName.isEmpty else {
            errorMessage = "Name is required."
            return
        }
        guard let gguf = URL(string: trimmedGGUF), gguf.scheme?.hasPrefix("http") == true else {
            errorMessage = "GGUF URL must be http(s)."
            return
        }

        var mmproj: URL?
        if modelKind == .vision {
            guard let mmprojURL = URL(string: trimmedMMProj), mmprojURL.scheme?.hasPrefix("http") == true else {
                errorMessage = "MMProj URL must be http(s)."
                return
            }
            mmproj = mmprojURL
        }

        let modelId = "custom-\(UUID().uuidString)"
        let filename = gguf.lastPathComponent.isEmpty ? "\(modelId).gguf" : gguf.lastPathComponent
        let projectorFilename = mmproj?.lastPathComponent

        let newModel = ModelSpec(
            id: modelId,
            displayName: trimmedName,
            filename: filename,
            downloadURL: gguf,
            template: .chatML(),
            isMultimodal: modelKind == .vision,
            projectorFilename: projectorFilename,
            projectorURL: mmproj
        )
        modelStore.addCustomModel(newModel)
        isPresented = false
    }
}

#Preview("ModelDownloadView") {
    ModelDownloadView()
        .preferredColorScheme(.dark)
        .padding()
        .background(Color("BackgroundColor"))
}
