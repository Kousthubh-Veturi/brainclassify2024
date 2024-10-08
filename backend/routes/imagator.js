const express = require('express');
const multer = require('multer');
const { PythonShell } = require('python-shell');
const History = require('../models/historyModel');  
const router = express.Router();


const upload = multer({ dest: 'uploads/' });


router.post('/upload', upload.single('image'), (req, res) => {
    const imagePath = req.file.path;


    const options = {
        mode: 'text',
        pythonOptions: ['-u'],
        args: [imagePath]       
    };


    PythonShell.run('./process_image.py', options, async (err, results) => {
        if (err) {
            console.error('Error processing image:', err);
            return res.status(500).json({ message: 'Error processing image' });
        }

     
        const classification = results[0];  
        const featureVectors = results[1]; 

        try {
            await History.create({
                userId: req.user.id,  
                featureVectors: JSON.stringify(featureVectors),  
                classification, 
            });


            res.status(200).json({ classification });
        } catch (err) {
            console.error('Database error:', err);
            res.status(500).json({ message: 'Error saving classification to database' });
        }
    });
});

module.exports = router;