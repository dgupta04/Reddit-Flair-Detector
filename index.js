const express = require('express');
const spawn = require('child_process').spawn
const app = express();
const port = process.env.PORT || 5000;

app.listen(port, ()=>{
    console.log("Server running at " + port);
})

app.use('/', express.static('public'));

app.get('/print', (req, res)=>{
    const childProcess = spawn('python', ['./worker/predictor.py']);
    res.set('Content-Type', 'text/plain');
    // mainRes = null;
    childProcess.stdout.pipe(res);
    // res.send();
})