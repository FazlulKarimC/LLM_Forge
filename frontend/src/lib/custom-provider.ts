export interface CustomProviderCredentials { customBaseUrl?: string; customApiKey?: string }
export interface CustomProviderRunConfig { provider?: "auto" | "hf_api" | "openrouter" | "groq" | "custom"; model_name?: string }

export function clearCustomProviderCredentials(): void {
    if (typeof window === "undefined") return;
    for (const storage of [window.sessionStorage, window.localStorage]) {
        for (const key of ["customLLMSettings", "customBaseUrl", "customApiKey", "customModelId"]) {
            try { storage.removeItem(key); } catch { /* Browser storage can be disabled. */ }
        }
    }
}

// Compatibility for existing imports: credentials are never restored or persisted.
export function getStoredCustomProviderCredentials(): CustomProviderCredentials { clearCustomProviderCredentials(); return {}; }
export function persistCustomProviderCredentials(): void { clearCustomProviderCredentials(); }
export function resolveRunExperimentCredentials(config?: CustomProviderRunConfig, credentials: CustomProviderCredentials = {}): CustomProviderCredentials {
    clearCustomProviderCredentials();
    if (config?.provider !== "custom") return {};
    if (!credentials.customBaseUrl) throw new Error("Enter endpoint credentials for this run. Credentials are kept only in this form and cleared when accepted.");
    return credentials;
}
