function openTab(evt, tabName) {
  const tabContents = document.querySelectorAll('.tab-content');
  const tabLinks = document.querySelectorAll('.tab-link');

  // Hide all tab contents
  // tabContents.forEach(content => content.style.display = 'none');
  tabContents.forEach(content => content.classList.remove('show-tab-content'));

  // Remove active class from all tabs
  tabLinks.forEach(link => link.classList.remove('active'));

  // Show the selected tab content
  // document.getElementById(tabName).style.display = 'block';
  evt.currentTarget.classList.add('active');
  document.getElementById(tabName).classList.add('show-tab-content');
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('validateuniprotid').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission
  
    const query = document.getElementById('uid').value;
    const url = `https://rest.uniprot.org/uniprotkb/search?size=1&query=${query}`;
  
    const errorDiv = document.getElementById('error');
    const errorMsg = errorDiv.querySelector('p');
    // errorDiv.style.display = 'none'; // Hide error message initially
    errorDiv.classList.add('show-error');
  
    fetch(url)
      .then(response => response.json())
      .then(data => {
        if (data.results && data.results.length > 0) {
            console.log('Results found:', data.results);
            // Proceed with form submission or further action
        } else {
            errorMsg.textContent = 'UniProt ID is invalid. Please check your query';
            // errorDiv.style.display = 'block'; // Show error message
        }
      })
      .catch(error => {
        errorMsg.textContent = 'Error occurred during the request. Please try again.';
        // errorDiv.style.display = 'block'; // Show error message
        console.error('Error:', error);
      });
  });
});
