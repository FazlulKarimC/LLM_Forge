export interface CustomProviderCredentials {
    customBaseUrl?: string;
    customApiKey?: string;
}

export interface CustomProviderRunConfig {
    provider?: "auto" | "hf_api" | "openrouter" | "groq" | "custom";
    model_name?: string;
}

const CUSTOM_LLM_SETTINGS_KEY = "customLLMSettings";
const CUSTOM_BASE_URL_KEY = "customBaseUrl";
const CUSTOM_API_KEY_KEY = "customApiKey";
const CUSTOM_MODEL_ID_KEY = "customModelId";

function getCustomProviderStorage(): Storage | null {
    if (typeof window === "undefined") {
        return null;
    }
    return window.sessionStorage;
}

export function getStoredCustomProviderCredentials(modelName?: string): CustomProviderCredentials {
    const storage = getCustomProviderStorage();
    if (!storage) {
        return {};
    }

    try {
        const settings = JSON.parse(storage.getItem(CUSTOM_LLM_SETTINGS_KEY) || "{}") as Record<
            string,
            { baseUrl?: string; apiKey?: string }
        >;

        if (modelName && settings?.[modelName]?.baseUrl) {
            return {
                customBaseUrl: settings[modelName].baseUrl,
                customApiKey: settings[modelName].apiKey,
            };
        }

        if (modelName && storage.getItem(CUSTOM_MODEL_ID_KEY) === modelName) {
            return {
                customBaseUrl: storage.getItem(CUSTOM_BASE_URL_KEY) || undefined,
                customApiKey: storage.getItem(CUSTOM_API_KEY_KEY) || undefined,
            };
        }
    } catch (error) {
        console.error("Failed to load custom model settings", error);
    }

    return {};
}

export function persistCustomProviderCredentials(
    modelName: string,
    customBaseUrl: string,
    customApiKey: string,
): void {
    const storage = getCustomProviderStorage();
    if (!storage || !modelName.trim()) {
        return;
    }

    try {
        const settings = JSON.parse(storage.getItem(CUSTOM_LLM_SETTINGS_KEY) || "{}") as Record<
            string,
            { baseUrl?: string; apiKey?: string }
        >;
        settings[modelName] = { baseUrl: customBaseUrl, apiKey: customApiKey };
        storage.setItem(CUSTOM_LLM_SETTINGS_KEY, JSON.stringify(settings));
        storage.setItem(CUSTOM_BASE_URL_KEY, customBaseUrl);
        storage.setItem(CUSTOM_API_KEY_KEY, customApiKey);
        storage.setItem(CUSTOM_MODEL_ID_KEY, modelName);
    } catch (error) {
        console.error("Failed to persist custom model settings", error);
    }
}

export function resolveRunExperimentCredentials(config?: CustomProviderRunConfig): CustomProviderCredentials {
    const credentials = getStoredCustomProviderCredentials(config?.model_name);

    if (config?.provider === "custom" && !credentials.customBaseUrl) {
        throw new Error(
            `No saved custom endpoint settings were found for model "${config.model_name}". Update the custom provider settings and try again.`,
        );
    }

    return credentials;
}
