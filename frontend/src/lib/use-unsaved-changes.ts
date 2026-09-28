"use client";

import { useEffect } from "react";

const warning = "You have unsaved changes. Leave this page and discard them?";

export function useUnsavedChanges(dirty: boolean) {
  useEffect(() => {
    if (!dirty) return;
    const beforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    const beforeWorkspaceSwitch = (event: Event) => {
      if (!window.confirm(warning)) event.preventDefault();
    };
    const beforeLinkNavigation = (event: MouseEvent) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const anchor = (event.target as Element).closest("a[href]");
      if (!anchor || anchor.hasAttribute("download") || anchor.getAttribute("target") === "_blank") return;
      const target = new URL(anchor.getAttribute("href")!, window.location.href);
      if (target.origin !== window.location.origin || target.href === window.location.href) return;
      if (!window.confirm(warning)) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    window.addEventListener("beforeunload", beforeUnload);
    window.addEventListener("llmforge:before-workspace-switch", beforeWorkspaceSwitch);
    document.addEventListener("click", beforeLinkNavigation, true);
    return () => {
      window.removeEventListener("beforeunload", beforeUnload);
      window.removeEventListener("llmforge:before-workspace-switch", beforeWorkspaceSwitch);
      document.removeEventListener("click", beforeLinkNavigation, true);
    };
  }, [dirty]);
}
