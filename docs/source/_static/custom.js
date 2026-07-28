"use strict";

document.addEventListener("DOMContentLoaded", () => {
    const html = document.documentElement;
    const themeButton = document.getElementById(
        "research-theme-toggle"
    );

    const validThemes = ["auto", "light", "dark"];

    const systemPreference = window.matchMedia(
        "(prefers-color-scheme: dark)"
    );

    const nextTheme = {
        auto: "light",
        light: "dark",
        dark: "auto",
    };

    const themeNames = {
        auto: "system",
        light: "light",
        dark: "dark",
    };

    const readStoredTheme = () => {
        try {
            const stored =
                window.localStorage.getItem("theme")
                || window.localStorage.getItem("mode");

            if (stored === "system") {
                return "auto";
            }

            return validThemes.includes(stored)
                ? stored
                : "auto";
        } catch (_) {
            return "auto";
        }
    };

    const storeTheme = (theme) => {
        try {
            /*
             * Store both keys for compatibility with different
             * PyData Sphinx Theme releases.
             */
            window.localStorage.setItem("theme", theme);
            window.localStorage.setItem("mode", theme);
        } catch (_) {
            // Continue without persistent storage.
        }
    };

    const applyTheme = (theme) => {
        const selected = validThemes.includes(theme)
            ? theme
            : "auto";

        /*
         * Resolve system mode to the exact same light or dark
         * value used by the manually selected modes.
         */
        const resolved = selected === "auto"
            ? (
                systemPreference.matches
                    ? "dark"
                    : "light"
            )
            : selected;

        html.setAttribute("data-theme", resolved);
        storeTheme(selected);

        if (!themeButton) {
            return;
        }

        themeButton.dataset.themeMode = selected;

        const currentName = themeNames[selected];
        const nextName = themeNames[nextTheme[selected]];

        themeButton.setAttribute(
            "aria-label",
            `Theme: ${currentName}. `
            + `Click to use ${nextName} theme.`
        );

        themeButton.setAttribute(
            "title",
            `Theme: ${currentName}`
        );
    };

    applyTheme(readStoredTheme());

    if (themeButton) {
        themeButton.addEventListener("click", () => {
            const current =
                themeButton.dataset.themeMode || "auto";

            applyTheme(nextTheme[current] || "auto");
        });
    }

    /*
     * If system mode is active, immediately follow an
     * operating-system appearance change.
     */
    systemPreference.addEventListener(
        "change",
        () => {
            if (
                themeButton
                && themeButton.dataset.themeMode === "auto"
            ) {
                applyTheme("auto");
            }
        }
    );

    /*
     * Responsive navigation drawer.
     */
    const sidebar = document.querySelector(
        ".bd-sidebar-primary"
    );

    if (!sidebar) {
        return;
    }

    const menuButton = document.createElement("button");
    menuButton.type = "button";
    menuButton.className = "mobile-navigation-button";
    menuButton.setAttribute(
        "aria-label",
        "Open documentation navigation"
    );
    menuButton.setAttribute("aria-expanded", "false");

    menuButton.innerHTML = `
        <svg aria-hidden="true" viewBox="0 0 24 24">
            <path d="M4 7H20"></path>
            <path d="M4 12H20"></path>
            <path d="M4 17H20"></path>
        </svg>
    `;

    const backdrop = document.createElement("button");
    backdrop.type = "button";
    backdrop.className = "mobile-navigation-backdrop";
    backdrop.setAttribute(
        "aria-label",
        "Close documentation navigation"
    );

    document.body.appendChild(menuButton);
    document.body.appendChild(backdrop);

    const setMenuOpen = (open) => {
        document.body.classList.toggle(
            "mobile-sidebar-open",
            open
        );

        menuButton.setAttribute(
            "aria-expanded",
            String(open)
        );

        menuButton.setAttribute(
            "aria-label",
            open
                ? "Close documentation navigation"
                : "Open documentation navigation"
        );
    };

    menuButton.addEventListener("click", () => {
        setMenuOpen(
            !document.body.classList.contains(
                "mobile-sidebar-open"
            )
        );
    });

    backdrop.addEventListener("click", () => {
        setMenuOpen(false);
    });

    sidebar.addEventListener("click", (event) => {
        if (
            event.target.closest(
                ".research-global-navigation a"
            )
        ) {
            setMenuOpen(false);
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            setMenuOpen(false);
        }
    });

    window.addEventListener("resize", () => {
        if (window.innerWidth > 1100) {
            setMenuOpen(false);
        }
    });
});
