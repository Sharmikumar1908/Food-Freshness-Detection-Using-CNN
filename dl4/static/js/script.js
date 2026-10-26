(() => {
  const form = document.querySelector("#upload-form");
  const input = document.querySelector("#image-input");
  const dropZone = document.querySelector("#drop-zone");
  const preview = document.querySelector("#image-preview");
  const uploadTitle = document.querySelector("#upload-title");
  const message = document.querySelector("#form-message");
  const submitButton = document.querySelector("#submit-button");

  const showFile = (file) => {
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      message.textContent = "Choose a JPG, PNG, or WebP image.";
      message.className = "form-message error";
      input.value = "";
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      message.textContent = "Image must be 8 MB or smaller.";
      message.className = "form-message error";
      input.value = "";
      return;
    }
    message.textContent = "";
    message.className = "form-message";
    uploadTitle.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  };

  if (input && dropZone) {
    input.addEventListener("change", () => showFile(input.files[0]));
    ["dragenter", "dragover"].forEach((eventName) =>
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-over");
      }),
    );
    ["dragleave", "drop"].forEach((eventName) =>
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
      }),
    );
    dropZone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (!file) return;
      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      showFile(file);
    });
  }

  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      message.textContent = "Analyzing your image…";
      message.className = "form-message";
      submitButton.disabled = true;
      submitButton.textContent = "Analyzing…";
      try {
        const response = await fetch(form.action, { method: "POST", body: new FormData(form) });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Prediction could not be completed.");
        const target = new URL(result.result_url, window.location.origin);
        target.searchParams.set("image", result.image_url);
        window.location.assign(target);
      } catch (error) {
        message.textContent = error.message || "Something went wrong. Please try again.";
        message.className = "form-message error";
        submitButton.disabled = false;
        submitButton.innerHTML = 'Analyze my food <span aria-hidden="true">→</span>';
      }
    });
  }

  const servingForm = document.querySelector("#serving-form");
  if (servingForm) {
    servingForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!servingForm.reportValidity()) return;
      const status = document.querySelector("#serving-message");
      const button = servingForm.querySelector("button");
      status.textContent = "Updating estimate…";
      status.className = "form-message";
      button.disabled = true;
      try {
        const response = await fetch(servingForm.dataset.url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ serving_size: document.querySelector("#result-serving").value }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Could not update serving quantity.");
        document.querySelector("#calorie-number").innerHTML =
          `${Math.round(result.estimated_calories)} <small>kcal</small>`;
        status.textContent = "Estimate updated.";
        status.className = "form-message success";
      } catch (error) {
        status.textContent = error.message || "Could not update estimate.";
        status.className = "form-message error";
      } finally {
        button.disabled = false;
      }
    });
  }
})();
