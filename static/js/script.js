const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const dropArea = document.getElementById('dropArea');
const loadingContainer=document.getElementById('loadingContainer');

dropArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file){
        return;
    }
    
    // Read file as image
    const reader=new FileReader();
    reader.onload=function(e){
        // Show preview
        preview.src=e.target.result;
        preview.style.display='block';

        // Hide upload text
        const uploadText=document.getElementById('uploadText');
        if (uploadText){
            uploadText.style.display='none';
        }
        
        dropArea.classList.add('hasImage');

        // Show loading animation
        loadingContainer.hidden = false;
        
         // Submit form after 3 seconds
        setTimeout(()=>{
            form.submit();
        },3000);
        
    };
    reader.readAsDataURL(file);
});

