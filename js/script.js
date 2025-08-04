const form=document.getElementById('uploadForm');
const result=document.getElementById('result');

form.addEventListener('submit',async(e)=>{
    e.preventDefault();
    const fileInput=document.getElementById('fileInput');
    const formData=new FormData();
    formData.append('file',fileInput.files[0]);

    try{
        const res =await fetch('/predict',{
            method:'POST',
            body: formData
        });

        const data=await res.json();
        result.textContent=data.species ? `Especie detectada: ${data.species}`  : `Error: ${data.error}`;
    } catch (err){
        result.textContent='ERROR';
    }
});