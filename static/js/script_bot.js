// Get the modal
var modal = document.getElementById("analysisModal");

// Get the button that opens the modal
var btn = document.getElementById("analyzeButton");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// Open the modal with a fade-in effect
btn.onclick = function() {
  modal.style.display = "block";
  modal.style.opacity = 0;
  setTimeout(() => modal.style.opacity = 1, 50);
}

// Close the modal with a fade-out effect
span.onclick = function() {
  modal.style.opacity = 0;
  setTimeout(() => modal.style.display = "none", 300);
}

// Close the modal when clicking outside of it
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.opacity = 0;
    setTimeout(() => modal.style.display = "none", 300);
  }
}
