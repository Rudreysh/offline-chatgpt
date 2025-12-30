import Foundation
import Combine

final class ModelStore: ObservableObject {
    static let shared = ModelStore()

    @Published private(set) var models: [ModelSpec] = []
    private var customModels: [ModelSpec] = []
    @Published var selectedModelID: String {
        didSet {
            UserDefaults.standard.set(selectedModelID, forKey: Self.selectedModelKey)
        }
    }

    private static let selectedModelKey = "selectedModelID"
    private static let customModelsKey = "customModels"

    private init() {
        let saved = UserDefaults.standard.string(forKey: Self.selectedModelKey)
        self.selectedModelID = saved ?? ModelCatalog.models.first?.id ?? "olmoe-latest"
        self.customModels = Self.loadCustomModels()
        self.models = ModelCatalog.models + customModels
        if !self.models.contains(where: { $0.id == selectedModelID }) {
            self.selectedModelID = self.models.first?.id ?? "olmoe-latest"
        }
    }

    var selectedModel: ModelSpec {
        models.first(where: { $0.id == selectedModelID }) ?? models[0]
    }

    func selectModel(_ model: ModelSpec) {
        selectedModelID = model.id
    }

    func isReady(_ model: ModelSpec) -> Bool {
        model.isModelDownloaded && (model.isMultimodal ? model.isProjectorDownloaded : true)
    }

    func addCustomModel(_ model: ModelSpec) {
        customModels.append(model)
        persistCustomModels()
        models = ModelCatalog.models + customModels
    }

    func isCustom(_ model: ModelSpec) -> Bool {
        customModels.contains(where: { $0.id == model.id })
    }

    func removeCustomModel(_ model: ModelSpec) {
        customModels.removeAll(where: { $0.id == model.id })
        persistCustomModels()
        models = ModelCatalog.models + customModels
        if selectedModelID == model.id {
            selectedModelID = models.first?.id ?? "olmoe-latest"
        }
    }

    private func persistCustomModels() {
        let payload = customModels.map { CustomModel(from: $0) }
        guard let data = try? JSONEncoder().encode(payload) else { return }
        UserDefaults.standard.set(data, forKey: Self.customModelsKey)
    }

    private static func loadCustomModels() -> [ModelSpec] {
        guard let data = UserDefaults.standard.data(forKey: Self.customModelsKey),
              let payload = try? JSONDecoder().decode([CustomModel].self, from: data) else {
            return []
        }
        return payload.map { $0.toModelSpec() }
    }
}

private struct CustomModel: Codable {
    enum TemplateKind: String, Codable {
        case chatML
    }

    let id: String
    let displayName: String
    let filename: String
    let downloadURL: String
    let template: TemplateKind
    let isMultimodal: Bool
    let projectorFilename: String?
    let projectorURL: String?

    init(from model: ModelSpec) {
        self.id = model.id
        self.displayName = model.displayName
        self.filename = model.filename
        self.downloadURL = model.downloadURL.absoluteString
        self.template = .chatML
        self.isMultimodal = model.isMultimodal
        self.projectorFilename = model.projectorFilename
        self.projectorURL = model.projectorURL?.absoluteString
    }

    func toModelSpec() -> ModelSpec {
        ModelSpec(
            id: id,
            displayName: displayName,
            filename: filename,
            downloadURL: URL(string: downloadURL)!,
            template: .chatML(),
            isMultimodal: isMultimodal,
            projectorFilename: projectorFilename,
            projectorURL: projectorURL.flatMap { URL(string: $0) }
        )
    }
}
