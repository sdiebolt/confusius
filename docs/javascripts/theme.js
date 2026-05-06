// Patch the navbar site title so "fUSI" renders in the accent color.
// The header does not re-render on instant navigation, so one run suffices.
(function () {
  function patchNavTitle() {
    const el = document.querySelector(
      ".md-header__topic:first-child .md-ellipsis"
    );
    if (el && el.textContent.trim() === "ConfUSIus" && !el.querySelector(".fusi-accent")) {
      el.innerHTML = 'Con<span class="fusi-accent">fUSI</span>us';
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", patchNavTitle);
  } else {
    patchNavTitle();
  }
})();
