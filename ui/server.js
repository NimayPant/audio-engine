const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static('public'));

const upload = multer({ dest: os.tmpdir() });

app.post('/api/process', upload.single('audio'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No audio file uploaded.');
    }

    const dspBinaryPath = path.resolve(__dirname, '../build/Debug/dsp.exe');
    const alternativeDspPath = path.resolve(__dirname, '../build/dsp.exe');
    
    let exePath = dspBinaryPath;
    if (fs.existsSync(alternativeDspPath)) {
        exePath = alternativeDspPath;
    }

    if (!fs.existsSync(exePath) && os.platform() === 'win32') {
         console.warn(`DSP binary not found at ${exePath}. Ensure you have compiled the C++ project.`);
    } else if (os.platform() !== 'win32') {
         exePath = path.resolve(__dirname, '../build/dsp');
    }

    const inputRaw = req.file.path;
    const origExt = path.extname(req.file.originalname || '').toLowerCase() || '.wav';
    const origBase = path.basename(req.file.originalname || 'audio', origExt);
    const inputFile = inputRaw + origExt;
    fs.renameSync(inputRaw, inputFile);
    const outputFile = path.join(os.tmpdir(), `processed_${Date.now()}${origExt}`);
    
    let effects = [];
    try {
        if (req.body.effects) {
            effects = JSON.parse(req.body.effects);
        }
    } catch (e) {
        console.error("Error parsing effects JSON", e);
        return res.status(400).send('Invalid effects format.');
    }

    const args = [inputFile, outputFile];
    
    for (const effect of effects) {
        args.push(`--${effect.id}`);
        for (const param of effect.params) {
            args.push(param.toString());
        }
    }

    console.log(`Executing: ${exePath} ${args.join(' ')}`);

    const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
    const ffmpegDir = path.dirname(ffmpegPath);
    const childEnv = Object.assign({}, process.env);
    const pathKey = Object.keys(childEnv).find(k => k.toLowerCase() === 'path') || 'PATH';
    childEnv[pathKey] = ffmpegDir + path.delimiter + childEnv[pathKey];

    const child = spawn(exePath, args, { env: childEnv });

    let stderr = '';

    child.stderr.on('data', (data) => {
        stderr += data.toString();
    });

    child.on('close', (code) => {
        fs.unlink(inputFile, () => {});

        if (code !== 0) {
            console.error(`DSP command failed with code ${code}. Stderr: ${stderr}`);
            console.log("Returning original file instead due to failure.");
            res.status(500).send('DSP Processing failed: ' + stderr);
            return;
        }

        const downloadName = `${origBase}_processed${origExt}`;
        res.setHeader('Content-Disposition', `attachment; filename="${downloadName}"`);
        res.setHeader('Content-Type', origExt === '.mp3' ? 'audio/mpeg' : 'audio/wav');
        res.sendFile(outputFile, (err) => {
            if (err) {
                console.error("Error sending file", err);
            }
            fs.unlink(outputFile, () => {});
        });
    });
    
    child.on('error', (err) => {
        console.error('Failed to start subprocess.', err);
        fs.unlink(inputFile, () => {});
        res.status(500).send('Error executing DSP application. Ensure it is compiled and accessible.');
    });
});

app.get('/api/status', (req, res) => {
    res.json({
        status: 'online',
        uptime: process.uptime(),
        engine: 'C++ DSP Engine (ML-Based)',
        platform: os.platform(),
        cpus: os.cpus().length,
        memoryUsage: process.memoryUsage(),
        features: ['Dynamic EQ', 'Genre Detection', 'ML Inference', 'Offline Batch Processing']
    });
});

app.post('/api/ml-process', upload.single('audio'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No audio file uploaded.');
    }

    const verifyCppPath = path.resolve(__dirname, '../build/Debug/verify_cpp.exe');
    const altCppPath = path.resolve(__dirname, '../build/verify_cpp.exe');
    
    let exePath = verifyCppPath;
    if (fs.existsSync(altCppPath)) {
        exePath = altCppPath;
    }

    if (!fs.existsSync(exePath) && os.platform() !== 'win32') {
        exePath = path.resolve(__dirname, '../build/verify_cpp');
    }

    const inputRaw = req.file.path;
    const origExt = path.extname(req.file.originalname).toLowerCase() || '.wav';
    const inputFile = inputRaw + origExt;
    fs.renameSync(inputRaw, inputFile);

    const child = spawn(exePath, [inputFile]);

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
        stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
        stderr += data.toString();
    });

    let responded = false;
    child.on('close', (code) => {
        if (responded) return;
        fs.unlink(inputFile, () => {});

        if (code !== 0) {
            responded = true;
            console.error(`ML inference failed with code ${code}`);
            return res.status(500).json({ error: 'ML inference failed', details: stderr });
        }

        try {
            const projectRoot = path.resolve(__dirname, '..');
            const outputPath = path.join(projectRoot, 'tools', 'extracted_features', 'cpp_inference_output.json');
            const results = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
            
            responded = true;
            res.json({
                status: 'success',
                ml_results: results,
                genre_id: results.genre_id,
                genre_name: results.genre_name,
                genre_confidence: results.genre_confidence,
                eq_gains: results.eq_gains,
                message: 'ML inference completed'
            });
        } catch (e) {
            responded = true;
            console.error('Error reading ML results:', e);
            res.status(500).json({ error: 'Error reading ML results', details: e.message });
        }
    });

    child.on('error', (err) => {
        if (responded) return;
        responded = true;
        console.error('Failed to run ML inference:', err);
        res.status(500).json({ error: 'Inference process error', details: err.message });
    });
});

app.post('/api/ml-full-process', upload.single('audio'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No audio file uploaded.');
    }

    const dspBinaryPath = path.resolve(__dirname, '../build/Debug/dsp.exe');
    const alternativeDspPath = path.resolve(__dirname, '../build/dsp.exe');
    
    let exePath = dspBinaryPath;
    if (fs.existsSync(alternativeDspPath)) {
        exePath = alternativeDspPath;
    }

    const inputFile = req.file.path;
    const outputFile = path.join(os.tmpdir(), `ml_processed_${Date.now()}.wav`);
    
    const mlEqIntensity = parseFloat(req.body.mlEqIntensity) || 0.5;
    const genreId = parseInt(req.body.genreId) || 0;
    const genreIntensity = parseFloat(req.body.genreIntensity) || 1.0;

    const args = [inputFile, outputFile, '--ml-dynamic-eq', mlEqIntensity.toString()];
    
    if (genreId > 0) {
        args.push('--genre-aware');
        args.push(genreId.toString());
        args.push(genreIntensity.toString());
    }

    console.log(`[ML] Executing: ${exePath} ${args.join(' ')}`);

    const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
    const ffmpegDir = path.dirname(ffmpegPath);
    const childEnv = Object.assign({}, process.env);
    const pathKey = Object.keys(childEnv).find(k => k.toLowerCase() === 'path') || 'PATH';
    childEnv[pathKey] = ffmpegDir + path.delimiter + childEnv[pathKey];

    const child = spawn(exePath, args, { env: childEnv });

    let stderr = '';

    child.stderr.on('data', (data) => {
        stderr += data.toString();
    });

    child.on('close', (code) => {
        fs.unlink(inputFile, () => {});

        if (code !== 0) {
            console.error(`ML DSP command failed with code ${code}`);
            return res.status(500).json({ 
                error: 'ML DSP Processing failed',
                details: stderr 
                
            });
        }

        res.json({
            status: 'success',
            message: 'ML-based DSP processing completed',
            outputFile: path.basename(outputFile),
            mlEqIntensity,
            genreId,
            genreIntensity
        });

        setTimeout(() => {
            fs.unlink(outputFile, () => {});
        }, 5000);
    });
    
    child.on('error', (err) => {
        console.error('Failed to start ML DSP subprocess:', err);
        fs.unlink(inputFile, () => {});
        res.status(500).json({ 
            error: 'Error executing ML DSP',
            details: err.message 
        });
    });
});

app.listen(port, () => {
    console.log(`DSP Server Master initialized on port ${port}`);
    console.log(`ML-Based Audio Engine Ready.`);
    console.log(`Features: Dynamic EQ | Genre Detection | Offline Batch Processing`);
});
