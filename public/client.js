fetch('/print').then((res)=>{
    res.text().then(stuff=>{
        body = document.getElementsByTagName('body')[0];
        txt = document.createElement('p');
        txt.style.color = 'red';
        txt.innerText = stuff.replace(/\n/g, '');
        body.appendChild(txt);
        // console.log(stuff.replace(/\n/g, ''));

    })
})