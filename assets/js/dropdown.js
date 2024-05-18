function showImage() {
    const selector = document.getElementById('imageSelector');
    const dynamicImage = document.getElementById('dynamicImage');
    const btnElement = document.getElementById('toggleAttention');

    // Get the data-img-path attribute of the selected option
    const imagePath = selector.options[selector.selectedIndex].getAttribute('data-img-path');

    // Set opacity to 0 and then back to 1 to trigger the transition
    dynamicImage.style.opacity = 0;
    setTimeout(() => {
        dynamicImage.src = imagePath;
        dynamicImage.style.opacity = 1;
    }, 1000); // Small delay to allow the opacity change to take effect

    btnElement.textContent = 'Show Model Attention';
}
