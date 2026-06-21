document.addEventListener('DOMContentLoaded', () => {
    const logOutput = document.getElementById('log-output');
    const ranges = document.querySelectorAll('input[type="range"]');

    const addLog = (msg, type = 'system') => {
        if (!logOutput) return;
        const item = document.createElement('div');
        item.className = `log-item ${type}`;
        item.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        logOutput.appendChild(item);
        logOutput.scrollTop = logOutput.scrollHeight;
    };

    const checkServerStatus = async () => {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                addLog(`Connected: ${data.engine}`, 'success');
            }
        } catch (e) {
            addLog('Backend offline.', 'error');
        }
    };
    checkServerStatus();

    let selectedFile = null;
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    const drawWaveform = (audioBuffer) => {
        const waveCanvas = document.getElementById('waveform-canvas');
        const emptyMsg = document.getElementById('no-data-msg');
        if (!waveCanvas) return;
        
        if (emptyMsg) emptyMsg.style.display = 'none';
        
        const ctx = waveCanvas.getContext('2d');
        const targetW = Math.floor(waveCanvas.offsetWidth * window.devicePixelRatio);
        const targetH = Math.floor(waveCanvas.offsetHeight * window.devicePixelRatio);
        
        if (waveCanvas.width !== targetW || waveCanvas.height !== targetH) {
            waveCanvas.width = targetW;
            waveCanvas.height = targetH;
        }

        const w = waveCanvas.width;
        const h = waveCanvas.height;
        const data = audioBuffer.getChannelData(0);
        const step = Math.ceil(data.length / w);
        
        ctx.clearRect(0, 0, w, h);
        ctx.strokeStyle = '#4b7bec';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for(let i=0; i<w; i++) {
            let min=1, max=-1;
            for(let j=0; j<step; j++) {
                const d = data[(i*step)+j];
                if(d < min) min = d; if(d > max) max = d;
            }
            ctx.moveTo(i, (1+min)*h/2); ctx.lineTo(i, (1+max)*h/2);
        }
        ctx.stroke();
    };

    window.drawEQCurve = () => {
        const canvas = document.getElementById('eq-graph-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const targetW = Math.floor(canvas.offsetWidth * window.devicePixelRatio);
        const targetH = Math.floor(canvas.offsetHeight * window.devicePixelRatio);
        
        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }
        
        const w = canvas.width;
        const h = canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        for(let i=1; i<10; i++) {
            const x = (i/10) * w;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();

        const freq = parseFloat(document.getElementById('eq-freq').value);
        const q = parseFloat(document.getElementById('eq-q').value);
        const gain = parseFloat(document.getElementById('eq-gain').value);
        
        ctx.strokeStyle = '#4b7bec';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const centerLog = Math.log10(freq);
        const bandwidth = 1.0 / q;
        
        for (let x = 0; x <= w; x++) {
            const f = 20 * Math.pow(1000, x / w);
            const logF = Math.log10(f);
            const dist = (logF - centerLog) / (bandwidth * 0.5);
            const response = gain * Math.exp(-0.5 * dist * dist);
            const y = h/2 - (response / 24) * (h/2.5);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    };

    window.drawBiquadCurve = () => {
        const canvas = document.getElementById('biquad-graph-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const targetW = Math.floor(canvas.offsetWidth * window.devicePixelRatio);
        const targetH = Math.floor(canvas.offsetHeight * window.devicePixelRatio);
        
        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }
        
        const w = canvas.width;
        const h = canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        for(let i=1; i<10; i++) {
            const x = (i/10) * w;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();

        const freq = parseFloat(document.getElementById('biquad-freq').value);
        const q = parseFloat(document.getElementById('biquad-q').value);
        const gain = parseFloat(document.getElementById('biquad-gain').value);
        
        ctx.strokeStyle = '#4b7bec';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const centerLog = Math.log10(freq);
        const bandwidth = 1.0 / q;
        
        for (let x = 0; x <= w; x++) {
            const f = 20 * Math.pow(1000, x / w);
            const logF = Math.log10(f);
            const dist = (logF - centerLog) / (bandwidth * 0.5);
            const response = gain * Math.exp(-0.5 * dist * dist);
            const y = h/2 - (response / 24) * (h/2.5);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    };

    window.currentMLGains = [0,0,0,0,0,0,0,0];
    window.drawMLEQCurve = (gains) => {
        const activeGains = gains || window.currentMLGains || [0,0,0,0,0,0,0,0];
        const canvas = document.getElementById('ml-eq-graph-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const targetW = Math.floor(canvas.offsetWidth * window.devicePixelRatio);
        const targetH = Math.floor(canvas.offsetHeight * window.devicePixelRatio);

        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }

        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
        
        ctx.strokeStyle = '#88c999';
        ctx.lineWidth = 1.5;
        ctx.shadowBlur = 6;
        ctx.shadowColor = '#88c999';
        
        const intensity = parseFloat(document.getElementById('ml-eq-intensity')?.value || 0.5);
        const rawPoints = [0, ...activeGains, 0];
        
        ctx.beginPath();
        for (let x = 0; x < w; x += 2) {
            const t = (x / w) * (rawPoints.length - 1);
            const i0 = Math.floor(t);
            const i1 = Math.min(rawPoints.length - 1, i0 + 1);
            const alpha = t - i0;
            
            const smoothAlpha = (1 - Math.cos(alpha * Math.PI)) / 2;
            const gain = (rawPoints[i0] * (1 - smoothAlpha) + rawPoints[i1] * smoothAlpha) * intensity;
            
            const y = h / 2 - (gain / 12) * (h / 2.8);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    };

    const fileInput = document.getElementById('audio-file');
    const processBtn = document.getElementById('btn-process');
    
    if (fileInput) {
        fileInput.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                document.getElementById('file-name').textContent = selectedFile.name;
                processBtn.disabled = false;
                addLog(`Loaded: ${selectedFile.name}`);
                
                const arrayBuffer = await selectedFile.arrayBuffer();
                const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
                drawWaveform(audioBuffer);

                addLog('Pre-scanning audio for ML features...', 'system');
                const formData = new FormData();
                formData.append('audio', selectedFile);
                try {
                    const res = await fetch('/api/ml-process', { method: 'POST', body: formData });
                    if (res.ok) {
                        const data = await res.json();
                        if (data.ml_results) {
                            const genre = data.genre_name || 'Unknown';
                            const conf = (data.genre_confidence * 100).toFixed(1);
                            document.getElementById('detected-genre').textContent = genre;
                            document.getElementById('genre-confidence').textContent = `${conf}%`;
                            
                            document.getElementById('genre-select').value = data.genre_id;
                            
                            window.currentMLGains = data.eq_gains || [0,0,0,0,0,0,0,0];
                            window.drawMLEQCurve(window.currentMLGains);
                            
                            addLog(`ML Pre-scan: ${genre} (${conf}%)`, 'success');
                        }
                    }
                } catch (err) {
                    console.error('Scan failed', err);
                }
            }
        });
    }

    const presetBtn = document.getElementById('btn-default-preset');
    if (presetBtn) {
        presetBtn.addEventListener('click', () => {
            ranges.forEach(r => {
                const defaultValue = r.getAttribute('value') || (r.max - r.min) / 2;
                r.value = defaultValue;
                const valDisplay = document.getElementById('val-' + r.id);
                if (valDisplay) {
                    let suffix = '';
                    if (r.id.includes('freq')) suffix = ' Hz';
                    else if (r.id.includes('gain')) suffix = ' dB';
                    else if (r.id.includes('mix') || r.id.includes('room')) suffix = ' %';
                    valDisplay.textContent = r.value + suffix;
                }
            });
            document.querySelectorAll('input[type="checkbox"]').forEach(c => c.checked = false);
            window.drawEQCurve();
            window.drawBiquadCurve();
            addLog('Parameters reset', 'info');
        });
    }

    if (processBtn) {
        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            const effects = [];
            const addEffect = (id, params) => {
                const el = document.getElementById(`enable-${id}`);
                if (el && el.checked) effects.push({ id, params });
            };
            const getVal = (id) => parseFloat(document.getElementById(id)?.value || 0);

            addEffect('eq', [getVal('eq-freq'), getVal('eq-q'), getVal('eq-gain')]);
            addEffect('biquad-simd', [getVal('biquad-freq'), getVal('biquad-q'), getVal('biquad-gain')]);
            addEffect('delay', [getVal('delay-time'), getVal('delay-feedback')]);
            addEffect('reverb', [getVal('reverb-room'), 0.5, getVal('reverb-mix')]);
            addEffect('gain-simd', [getVal('gain-simd-val')]);
            addEffect('smart-compress', [getVal('smart-compress-intensity')]);
            addEffect('smart-gate', [getVal('smart-gate-thresh')]);
            addEffect('ml-dynamic-eq', [getVal('ml-eq-intensity')]);
            addEffect('genre-aware', [getVal('genre-select'), getVal('genre-intensity')]);

            processBtn.disabled = true;
            document.getElementById('spinner').classList.remove('hidden');
            addLog('Processing buffer...', 'system');
            
            try {
                const formData = new FormData();
                formData.append('audio', selectedFile);
                formData.append('effects', JSON.stringify(effects));
                const res = await fetch('/api/process', { method: 'POST', body: formData });
                if (!res.ok) throw new Error(await res.text());

                const contentDisposition = res.headers.get('Content-Disposition');
                let fileName = 'processed_audio.wav';
                if (contentDisposition) {
                    const match = contentDisposition.match(/filename="(.+)"/);
                    if (match && match[1]) fileName = match[1];
                }

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                document.getElementById('output-audio').src = url;
                
                const downloadLink = document.getElementById('download-link');
                downloadLink.href = url;
                downloadLink.download = fileName;
                
                document.getElementById('audio-player-container').classList.remove('hidden');
                addLog('Processing complete', 'success');
                const processedBuffer = await audioCtx.decodeAudioData(await blob.arrayBuffer());
                drawWaveform(processedBuffer);
            } catch (err) {
                addLog('Failure: ' + err.message, 'error');
            } finally {
                processBtn.disabled = false;
                document.getElementById('spinner').classList.add('hidden');
            }
        });
    }
    const genreSelect = document.getElementById('genre-select');
    if (genreSelect) {
        genreSelect.addEventListener('change', (e) => {
            const genreName = e.target.options[e.target.selectedIndex].text;
            addLog(`Selected genre: ${genreName}`, 'info');
        });
    }

    const playBtn = document.getElementById('btn-play-output');
    const audioEl = document.getElementById('output-audio');
    const progressFill = document.getElementById('progress-fill');

    if (playBtn && audioEl) {
        playBtn.addEventListener('click', () => {
            audioEl.play();
        });

        const pauseBtn = document.getElementById('btn-pause-output');
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => {
                audioEl.pause();
            });
        }

        audioEl.addEventListener('timeupdate', () => {
            const pct = (audioEl.currentTime / audioEl.duration) * 100;
            if (progressFill) progressFill.style.width = pct + '%';
        });

        audioEl.addEventListener('ended', () => {
            playBtn.textContent = 'PLAY';
            if (progressFill) progressFill.style.width = '0%';
        });
    }

    ranges.forEach(r => {
        r.addEventListener('input', () => {
            const valDisplay = document.getElementById('val-' + r.id);
            if (valDisplay) {
                let suffix = '';
                if (r.id.includes('freq')) suffix = ' Hz';
                else if (r.id.includes('gain')) suffix = ' dB';
                const isPercent = r.id.includes('mix') || r.id.includes('room') || r.id.includes('intensity');
                if (isPercent) suffix = ' %';
                valDisplay.textContent = (isPercent ? Math.round(r.value * 100) : r.value) + suffix;
            }
            if (r.id.startsWith('eq-')) window.drawEQCurve();
            if (r.id.startsWith('biquad-')) window.drawBiquadCurve();
            if (r.id.includes('ml-eq-intensity')) window.drawMLEQCurve(window.currentMLGains);
        });
    });

    window.drawEQCurve();
    window.drawBiquadCurve();
    window.drawMLEQCurve();
});
