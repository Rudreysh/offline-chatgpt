//
//  ContentView.swift
//  OLMoE.swift
//
//  Created by Luca Soldaini on 2024-09-16.
//


import SwiftUI
import UIKit
import PhotosUI
import UniformTypeIdentifiers
import os

class Bot: LLM {
    let modelSpec: ModelSpec

    init(model: ModelSpec) {
        self.modelSpec = model
        let deviceName = UIDevice.current.model
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "MMMM d, yyyy"
        let currentDate = dateFormatter.string(from: Date())

        let timeFormatter = DateFormatter()
        timeFormatter.dateFormat = "h:mm a"
        let currentTime = timeFormatter.string(from: Date())

        let systemPrompt = "You are OLMoE (Open Language Mixture of Expert), a small language model running on \(deviceName). You have been developed at the Allen Institute for AI (Ai2) in Seattle, WA, USA. Today is \(currentDate). The time is \(currentTime)."

        guard FileManager.default.fileExists(atPath: model.localModelURL.path) else {
            fatalError("Model file not found. Please download it first.")
        }

        let maxTokenCount: Int32 = model.isMultimodal ? 8192 : 2048
        super.init(
            from: model.localModelURL.path,
            stopSequence: model.template.stopSequence,
            maxTokenCount: maxTokenCount
        )
        self.template = model.template
    }
}

struct BotView: View {
    @StateObject var bot: Bot
    @State var input = ""
    @State private var isGenerating = false
    @State private var stopSubmitted = false
    @State private var scrollToBottom = false
    @State private var isSharing = false
    @State private var shareURL: URL?
    @State private var showShareSheet = false
    @State private var isSharingConfirmationVisible = false
    @State private var isDeleteHistoryConfirmationVisible = false
    @State private var isScrolledToBottom = true
    @FocusState private var isTextEditorFocused: Bool
    @Binding var showMetrics: Bool
    let disclaimerHandlers: DisclaimerHandlers
    let onBack: () -> Void

    // Add new state for text sharing
    @State private var showTextShareSheet = false
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var selectedFiles: [FileAttachment] = []
    @State private var showFileImporter = false
    @State private var showCamera = false
    @State private var attachmentErrorMessage: String?

    private var hasValidInput: Bool {
        !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || selectedImage != nil || !selectedFiles.isEmpty
    }

    private var isInputDisabled: Bool {
        isGenerating || isSharing
    }

    private var isDeleteButtonDisabled: Bool {
        isInputDisabled || bot.history.isEmpty
    }

    private var isChatEmpty: Bool {
        bot.history.isEmpty && !isGenerating && bot.output.isEmpty
    }

    private var isMtmdAvailable: Bool {
        MTMDRuntime.isAvailable
    }

    private var projectorExists: Bool {
        guard bot.modelSpec.isMultimodal else { return true }
        return bot.modelSpec.localProjectorURL?.exists ?? false
    }

    private var attachmentsEnabled: Bool {
        isMtmdAvailable && bot.modelSpec.isMultimodal && projectorExists
    }

    init(_ bot: Bot, showMetrics: Binding<Bool>, disclaimerHandlers: DisclaimerHandlers, onBack: @escaping () -> Void) {
        _bot = StateObject(wrappedValue: bot)
        _showMetrics = showMetrics
        self.disclaimerHandlers = disclaimerHandlers
        self.onBack = onBack
    }

    func shouldShowScrollButton() -> Bool {
        return !isScrolledToBottom
    }

    func respond() {
        isGenerating = true
        #if targetEnvironment(macCatalyst)
            isTextEditorFocused = true
        #else
            isTextEditorFocused = false
        #endif
        stopSubmitted = false
        var originalInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if originalInput.isEmpty {
            if selectedImage != nil {
                originalInput = "Describe the image."
            } else if !selectedFiles.isEmpty {
                originalInput = "Please review the attached files."
            }
        }
        if !selectedFiles.isEmpty {
            let fileList = selectedFiles.map { $0.name }.joined(separator: ", ")
            originalInput += "\n\nAttached files: \(fileList)"
        }
        input = "" // Clear the input after sending

        // Add the user message to history immediately
        bot.history.append(Chat(role: .user, content: originalInput))
        Task {
            if let image = selectedImage {
                guard attachmentsEnabled else {
                    await MainActor.run {
                        if !bot.modelSpec.isMultimodal {
                            attachmentErrorMessage = "Selected model does not support images."
                        } else if !isMtmdAvailable {
                            attachmentErrorMessage = "Vision runtime not available (mtmd not linked)."
                        } else {
                            attachmentErrorMessage = "Projector file (mmproj) missing for this model."
                        }
                        isGenerating = false
                    }
                    return
                }
                await bot.respond(to: originalInput, image: image, projectorURL: bot.modelSpec.localProjectorURL)
                await MainActor.run {
                    selectedImage = nil
                    selectedFiles.removeAll()
                }
            } else {
                await bot.respond(to: originalInput)
                await MainActor.run { selectedFiles.removeAll() }
            }
            await MainActor.run {
                bot.setOutput(to: "")
                isGenerating = false
                stopSubmitted = false
                #if targetEnvironment(macCatalyst)
                    isTextEditorFocused = true  // Mac Only. Re-focus after response
                #endif
            }
        }
    }

    func stop() {
        self.stopSubmitted = true
        Task { await bot.stop() }
    }

    func deleteHistory() {
        Task { @MainActor in
            await bot.clearHistory()
            bot.setOutput(to: "")
            input = "" // Clear the input
            // Reset metrics when clearing chat history
            bot.metrics.reset()
        }
    }

    private func formatConversationForSharing() -> String {
        let deviceName = UIDevice.current.model
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "MMMM d, yyyy 'at' h:mm a"
        let timestamp = dateFormatter.string(from: Date())

        let header = """
        Conversation with OLMoE (Open Language Mixture of Expert)
        ----------------------------------------

        """

        let conversation = bot.history.map { chat in
            let role = chat.role == .user ? "User" : "OLMoE"
            return "\(role): \(chat.content)"
        }.joined(separator: "\n\n")

        let footer = """

        ----------------------------------------
        Shared from OLMoE - AI2's Open Language Model
        https://github.com/allenai/OLMoE
        """

        return header + conversation + footer
    }

    func shareConversation() {
        isSharing = true
        disclaimerHandlers.setShowDisclaimerPage(false)
        Task {
            do {
                let attestationResult = try await AppAttestManager.performAttest()

                // Prepare payload
                let apiKey = Configuration.apiKey
                let apiUrl = Configuration.apiUrl

                let modelName = bot.modelSpec.filename
                let systemFingerprint = "\(modelName)-\(AppInfo.shared.appId)"

                let messages = bot.history.map { chat in
                    ["role": chat.role == .user ? "user" : "assistant", "content": chat.content]
                }

                let payload: [String: Any] = [
                    "model": modelName,
                    "system_fingerprint": systemFingerprint,
                    "created": Int(Date().timeIntervalSince1970),
                    "messages": messages,
                    "key_id": attestationResult.keyID,
                    "attestation_object": attestationResult.attestationObjectBase64
                ]

                let jsonData = try JSONSerialization.data(withJSONObject: payload)

                guard let url = URL(string: apiUrl), !apiUrl.isEmpty else {
                    print("Invalid URL")
                    await MainActor.run {
                        isSharing = false
                    }
                    return
                }

                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
                request.httpBody = jsonData
                let (data, response) = try await URLSession.shared.data(for: request)

                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let responseString = String(data: data, encoding: .utf8)!
                    if let jsonData = responseString.data(using: .utf8),
                       let jsonResult = try JSONSerialization.jsonObject(with: jsonData, options: []) as? [String: Any],
                       let body = jsonResult["body"] as? String,
                       let bodyData = body.data(using: .utf8),
                       let bodyJson = try JSONSerialization.jsonObject(with: bodyData, options: []) as? [String: Any],
                       let urlString = bodyJson["url"] as? String,
                       let url = URL(string: urlString) {
                        await MainActor.run {
                            self.shareURL = url
                            self.showShareSheet = true
                        }
                        print("Conversation shared successfully")
                    } else {
                        print("Failed to parse response")
                    }
                } else {
                    print("Failed to share conversation")
                }
            } catch {
                let attestError = error as NSError
                if attestError.domain == "AppAttest" {
                    print("Error: \(attestError.localizedDescription)")
                } else {
                    print("Error sharing conversation: \(error)")
                }
            }

            await MainActor.run {
                isSharing = false
            }
        }
    }

    @ViewBuilder
    func shareButton() -> some View {
        if isSharing {
            SpinnerView(color: Color("AccentColor"))
        } else {
            let isDisabled = isSharing || bot.history.isEmpty || isGenerating
            ToolbarButton(action: {
                isTextEditorFocused = false
                // disclaimerHandlers.setActiveDisclaimer(Disclaimers.ShareDisclaimer())
                // disclaimerHandlers.setCancelAction({ disclaimerHandlers.setShowDisclaimerPage(false) })
                // disclaimerHandlers.setAllowOutsideTapDismiss(true)
                // disclaimerHandlers.setConfirmAction({ shareConversation() })
                // disclaimerHandlers.setShowDisclaimerPage(true)
                showTextShareSheet = true
            }, assetName: "ShareIcon", foregroundColor: Color("AccentColor"))
             .disabled(isDisabled)
        }
    }

    @ViewBuilder
    func newChatButton() -> some View {
        ToolbarButton(action: {
            isTextEditorFocused = false
            isDeleteHistoryConfirmationVisible = true
            stop()
        }, assetName: "NewChatIcon", foregroundColor: Color("LightGreen"))
            .alert("Clear chat history?", isPresented: $isDeleteHistoryConfirmationVisible, actions: {
                Button("Clear", action: deleteHistory)
                Button("Cancel", role: .cancel) {
                    isDeleteHistoryConfirmationVisible = false
                }
            })
            .disabled(isDeleteButtonDisabled)
    }

    var body: some View {
        GeometryReader { geometry in
            contentView(in: geometry)
        }
    }

    @ViewBuilder
    private func contentView(in geometry: GeometryProxy) -> some View {
        ZStack {
            Color("BackgroundColor")
                .edgesIgnoringSafeArea(.all)

            VStack(alignment: .leading) {
                headerView()

                if !isChatEmpty {
                    ScrollViewReader { proxy in
                        ZStack {
                            ChatView(
                                history: bot.history,
                                output: bot.output.trimmingCharacters(in: .whitespacesAndNewlines),
                                metrics: bot.metrics,
                                showMetrics: $showMetrics,
                                isGenerating: $isGenerating,
                                isScrolledToBottom: $isScrolledToBottom,
                                stopSubmitted: $stopSubmitted
                            )
                                .onChange(of: scrollToBottom) { _, newValue in
                                    if newValue {
                                        withAnimation {
                                            proxy.scrollTo(ChatView.BottomID, anchor: .bottom)
                                        }
                                        scrollToBottom = false
                                    }
                                }
                                .gesture(TapGesture().onEnded({
                                    isTextEditorFocused = false
                                }))

                            ScrollToBottomButtonView(
                                scrollToBottom: $scrollToBottom,
                                shouldShowScrollButton: shouldShowScrollButton
                            )
                        }
                    }
                } else {
                    ZStack {
                        VStack{
                            Spacer()
                            Image("Ai2Icon")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: min(geometry.size.width, geometry.size.height) * 0.18)
                            Spacer()
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }

                Spacer()

                if (isChatEmpty) {
                    BotChatBubble(
                        text: String(localized: "Welcome chat message", comment: "Default chat bubble when conversation is empty"),
                        maxWidth: geometry.size.width,
                        hideCopyButton: true
                    )
                    .padding(.bottom, 15)
                }

                attachmentsView()

                inputBarView()
            }
            .padding(12)
        }
        .sheet(isPresented: $showShareSheet) {
            if let url = shareURL {
                ActivityViewController(activityItems: [url])
            }
        }
        .sheet(isPresented: $showTextShareSheet) {
            ActivityViewController(activityItems: [formatConversationForSharing()])
        }
        .sheet(isPresented: $showCamera) {
            CameraPicker(image: $selectedImage)
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [
                .pdf,
                .plainText,
                .rtf,
                .json,
                .xml,
                .commaSeparatedText,
                .data
            ],
            allowsMultipleSelection: true
        ) { result in
            handleFileImport(result)
        }
        .alert("Attachment Error", isPresented: Binding(get: {
            attachmentErrorMessage != nil
        }, set: { newValue in
            if !newValue { attachmentErrorMessage = nil }
        })) {
            Button("OK", role: .cancel) {
                attachmentErrorMessage = nil
            }
        } message: {
            Text(attachmentErrorMessage ?? "")
        }
        .gesture(TapGesture().onEnded({
            isTextEditorFocused = false
        }))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                #if targetEnvironment(macCatalyst)
                    let spacing: CGFloat = 20
                #else
                    let spacing: CGFloat = 32
                #endif
                HStack(alignment: .bottom, spacing: spacing) {
                    shareButton()
                    newChatButton()
                }
            }
        }
    }

    private func headerView() -> some View {
        HStack(spacing: 12) {
            Button(action: onBack) {
                Image(systemName: "chevron.left")
                    .foregroundColor(Color("TextColor"))
                    .padding(8)
                    .background(Color("Surface"))
                    .clipShape(Circle())
            }
            Text(bot.modelSpec.displayName)
                .font(.headline)
                .foregroundColor(Color("TextColor"))
            Spacer()
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    @ViewBuilder
    private func attachmentsView() -> some View {
        if !isTextEditorFocused {
            if let image = selectedImage {
                HStack(spacing: 12) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 72, height: 72)
                        .clipped()
                        .cornerRadius(12)
                    Text("Image attached")
                        .foregroundColor(Color("TextColor"))
                    Spacer()
                    Button("Remove") {
                        selectedImage = nil
                    }
                    .buttonStyle(.SecondaryButton)
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }
            if !selectedFiles.isEmpty {
                VStack(spacing: 8) {
                    ForEach(selectedFiles) { file in
                        HStack(spacing: 12) {
                            Image(systemName: file.iconName)
                                .foregroundColor(Color("LightGreen"))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(file.name)
                                    .foregroundColor(Color("TextColor"))
                                Text(file.detailText)
                                    .font(.caption)
                                    .foregroundColor(Color("TextColor").opacity(0.6))
                            }
                            Spacer()
                            Button("Remove") {
                                selectedFiles.removeAll { $0.id == file.id }
                            }
                            .buttonStyle(.SecondaryButton)
                        }
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color("Surface"))
                        )
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }
        }
    }

    private func inputBarView() -> some View {
        HStack(alignment: .bottom, spacing: 12) {
            if !isTextEditorFocused {
                HStack(spacing: 8) {
                    PhotosPicker(selection: $selectedPhotoItem, matching: .images, photoLibrary: .shared()) {
                        Image(systemName: "paperclip")
                            .font(.system(size: 18, weight: .medium))
                            .foregroundColor(attachmentsEnabled ? Color("LightGreen") : Color("TextColor").opacity(0.4))
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color("Surface"))
                            )
                    }
                    .disabled(!attachmentsEnabled)
                    .onChange(of: selectedPhotoItem) { _, newItem in
                        handlePhotoSelection(newItem)
                    }

                    Button {
                        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
                            attachmentErrorMessage = "Camera is not available on this device."
                            return
                        }
                        showCamera = true
                    } label: {
                        Image(systemName: "camera")
                            .font(.system(size: 18, weight: .medium))
                            .foregroundColor(attachmentsEnabled ? Color("LightGreen") : Color("TextColor").opacity(0.4))
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color("Surface"))
                            )
                    }
                    .disabled(!attachmentsEnabled)

                    Button {
                        showFileImporter = true
                    } label: {
                        Image(systemName: "doc")
                            .font(.system(size: 18, weight: .medium))
                            .foregroundColor(Color("LightGreen"))
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color("Surface"))
                            )
                    }
                }
            }

            MessageInputView(
                input: $input,
                isGenerating: $isGenerating,
                stopSubmitted: $stopSubmitted,
                isTextEditorFocused: $isTextEditorFocused,
                isInputDisabled: isInputDisabled,
                hasValidInput: hasValidInput,
                respond: respond,
                stop: stop
            )
        }
    }

    private func handlePhotoSelection(_ item: PhotosPickerItem?) {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    selectedImage = image
                }
            }
        }
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            let newFiles = urls.compactMap { url in
                storeAttachment(from: url)
            }
            selectedFiles.append(contentsOf: newFiles)
        case .failure(let error):
            attachmentErrorMessage = "Failed to import file: \(error.localizedDescription)"
        }
    }

    private func storeAttachment(from url: URL) -> FileAttachment? {
        let accessGranted = url.startAccessingSecurityScopedResource()
        defer {
            if accessGranted {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let fileName = url.lastPathComponent
        let destinationFolder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Attachments", isDirectory: true)
        let destinationURL = destinationFolder.appendingPathComponent(fileName)

        do {
            try FileManager.default.createDirectory(at: destinationFolder, withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: url, to: destinationURL)
            let attributes = try FileManager.default.attributesOfItem(atPath: destinationURL.path)
            let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            return FileAttachment(
                url: destinationURL,
                name: fileName,
                size: size,
                typeIdentifier: UTType(filenameExtension: destinationURL.pathExtension)?.identifier
            )
        } catch {
            attachmentErrorMessage = "Failed to save file: \(error.localizedDescription)"
            return nil
        }
    }
}

private struct FileAttachment: Identifiable, Equatable {
    let id = UUID()
    let url: URL
    let name: String
    let size: Int64
    let typeIdentifier: String?

    var iconName: String {
        if let typeIdentifier, UTType(typeIdentifier)?.conforms(to: .pdf) == true {
            return "doc.richtext"
        }
        if let typeIdentifier, UTType(typeIdentifier)?.conforms(to: .text) == true {
            return "doc.plaintext"
        }
        return "doc"
    }

    var detailText: String {
        let sizeText = ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
        return sizeText
    }
}

private struct CameraPicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    typealias UIViewControllerType = UIImagePickerController

    func makeUIViewController(context: UIViewControllerRepresentableContext<CameraPicker>) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: UIViewControllerRepresentableContext<CameraPicker>) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(image: $image)
    }

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        @Binding var image: UIImage?

        init(image: Binding<UIImage?>) {
            _image = image
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let selectedImage = info[.originalImage] as? UIImage {
                image = selectedImage
            }
            picker.dismiss(animated: true)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}

struct ViewHeightKey: PreferenceKey {
    static var defaultValue: CGFloat { 0 }
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}


// Add this struct to handle the UIActivityViewController
struct ActivityViewController: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: UIViewControllerRepresentableContext<ActivityViewController>) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: UIViewControllerRepresentableContext<ActivityViewController>) {}
}

struct ContentView: View {
    /// A shared instance of the background download manager.
    @StateObject private var downloadManager = BackgroundDownloadManager.shared
    @StateObject private var modelStore = ModelStore.shared

    /// The state of the disclaimer handling.
    @StateObject private var disclaimerState = DisclaimerState()

    /// The bot instance used for conversation.
    @State private var bot: Bot?

    /// A flag indicating whether to show the info page.
    @State private var showInfoPage: Bool = false

    /// A flag indicating whether the device is supported.
    @State private var isSupportedDevice: Bool = isDeviceSupported()

    /// A flag indicating whether to use mocked model responses.
    @State private var useMockedModelResponse: Bool = false

    /// A flag indicating whether to show metrics.
    @State private var showMetrics: Bool = false
    @State private var showModelList: Bool = true
    @State private var modelLoadError: String?

    /// Logger for tracking events in the ContentView.
    let logger = Logger(subsystem: "com.allenai.olmoe", category: "ContentView")

    public var body: some View {
        ZStack {
            NavigationStack {
                VStack {
                    if !isSupportedDevice && !useMockedModelResponse {
                        UnsupportedDeviceView(
                            proceedAnyway: { isSupportedDevice = true },
                            proceedMocked: {
                                bot?.loopBackTestResponse = true
                                useMockedModelResponse = true
                            }
                        )
                    } else if showModelList {
                        ModelDownloadView(onSelectModel: { _ in
                            showModelList = false
                            checkModelAndInitializeBot()
                        })
                    } else if modelStore.isReady(modelStore.selectedModel), let activeBot = bot {
                        BotView(
                            activeBot,
                            showMetrics: $showMetrics,
                            disclaimerHandlers: DisclaimerHandlers(
                            setActiveDisclaimer: { self.disclaimerState.activeDisclaimer = $0 },
                            setAllowOutsideTapDismiss: { self.disclaimerState.allowOutsideTapDismiss = $0 },
                            setCancelAction: { self.disclaimerState.onCancel = $0 },
                            setConfirmAction: { self.disclaimerState.onConfirm = $0 },
                            setShowDisclaimerPage: { self.disclaimerState.showDisclaimerPage = $0 }
                            ),
                            onBack: {
                                showModelList = true
                                self.bot = nil
                            }
                        )
                    } else {
                        ModelDownloadView(onSelectModel: { _ in
                            showModelList = false
                            checkModelAndInitializeBot()
                        })
                    }
                }
                .onChange(of: modelStore.selectedModelID) { _, _ in
                    checkModelAndInitializeBot()
                }
                .onChange(of: downloadManager.isDownloading) { _, _ in
                    checkModelAndInitializeBot()
            }
            .onAppear { checkModelAndInitializeBot() }
            .navigationBarTitleDisplayMode(.inline)
            .alert("Model Error", isPresented: Binding(get: {
                modelLoadError != nil
            }, set: { newValue in
                if !newValue { modelLoadError = nil }
            })) {
                Button("OK", role: .cancel) {
                    modelLoadError = nil
                }
            } message: {
                Text(modelLoadError ?? "")
            }
            .toolbar {
                AppToolbar(
                    leadingContent: {
                        HStack(alignment: .bottom, spacing: 16) {
                            if !showModelList, bot != nil {
                                    ToolbarButton(
                                        action: {
                                            showModelList = true
                                            bot = nil
                                        },
                                        systemName: "chevron.left",
                                        foregroundColor: Color("TextColor")
                                    )

                                    Menu {
                                        ForEach(modelStore.models.filter { modelStore.isReady($0) }) { model in
                                            Button(model.displayName) {
                                                modelStore.selectModel(model)
                                                showModelList = false
                                                checkModelAndInitializeBot()
                                            }
                                        }
                                    } label: {
                                        HStack(spacing: 6) {
                                            Text(modelStore.selectedModel.displayName)
                                                .font(.caption)
                                                .foregroundColor(Color("TextColor"))
                                            Image(systemName: "chevron.down")
                                                .font(.caption2)
                                                .foregroundColor(Color("TextColor"))
                                        }
                                    }
                                }

                                InfoButton(action: { showInfoPage = true })

                                MetricsButton(
                                    action: { showMetrics.toggle() },
                                    isShowing: showMetrics
                                )
                            }
                        }
                    )
                }
            }
            .onAppear {
                disclaimerState.showInitialDisclaimer()
            }
            .onChange(of: disclaimerState.showDisclaimerPage) { _, newValue in
                if !newValue {
                    showModelList = true
                }
            }
            .sheet(isPresented: $showInfoPage) {
                SheetWrapper {
                    InfoView(isPresented: $showInfoPage)
                }
            }
            .sheet(isPresented: $disclaimerState.showDisclaimerPage) {
                SheetWrapper {
                    DisclaimerPage(
                        message: disclaimerState.activeDisclaimer?.text ?? "",
                        title: disclaimerState.activeDisclaimer?.title ?? "",
                        titleText: disclaimerState.activeDisclaimer?.headerTextContent ?? [],
                        confirm: DisclaimerPage.PageButton(
                            text: disclaimerState.activeDisclaimer?.buttonText ?? "",
                            onTap: {
                                disclaimerState.onConfirm?()
                            }
                        ),
                        cancel: disclaimerState.onCancel.map { cancelAction in
                            DisclaimerPage.PageButton(
                                text: "Cancel",
                                onTap: {
                                    cancelAction()
                                    disclaimerState.activeDisclaimer = nil
                                }
                            )
                        }
                    )
                }
                .interactiveDismissDisabled(!disclaimerState.allowOutsideTapDismiss)
            }
        }
    }

    /// Checks if the model exists before initializing the bot
    private func checkModelAndInitializeBot() {
        let model = modelStore.selectedModel
        if modelStore.isReady(model) {
            initializeBot()
        } else {
            bot = nil
        }
    }

    /// Initializes the bot instance and sets the loopback test response flag.
    private func initializeBot() {
        let model = modelStore.selectedModel
        if let error = LLM.validateModel(at: model.localModelURL.path) {
            modelLoadError = error
            bot = nil
            return
        }
        bot = Bot(model: model)
        bot?.loopBackTestResponse = useMockedModelResponse
    }
}
