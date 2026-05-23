// --- Переключение вкладок ---
const tabs = document.querySelectorAll('.tab-btn');
const contents = document.querySelectorAll('.tab-content');
tabs.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        contents.forEach(c => c.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        tabs.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// --- Калькулятор МСА ---
async function calculateISA() {
    const h = document.getElementById('height_calc').value;
    const res = await fetch(`/isa/calc?height=${h}`);
    const data = await res.json();
    document.getElementById('isa_T').innerText = data.temperature_K;
    document.getElementById('isa_P').innerText = data.pressure_Pa;
    document.getElementById('isa_rho').innerText = data.density_kg_m3;
    document.getElementById('isa_a').innerText = data.speed_of_sound_m_s;
}

const calcBtn = document.getElementById('calc_isa_btn');
if (calcBtn) {
    calcBtn.addEventListener('click', calculateISA);
}
calculateISA();

// --- График температуры ---
async function loadTemperatureChart() {
    const heights = [];
    const temps = [];
    for (let h = 0; h <= 20000; h += 1000) {
        const res = await fetch(`/isa/calc?height=${h}`);
        const data = await res.json();
        heights.push(h);
        temps.push(data.temperature_K);
    }
    const ctx = document.getElementById('tempChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: { labels: heights, datasets: [{ label: 'Температура по МСА', data: temps, borderColor: '#3b82f6', fill: false }] },
        options: { responsive: true, plugins: { legend: { labels: { color: '#e2e8f0' } } } }
    });
}
loadTemperatureChart();

// --- ИИ обучение ---
document.addEventListener('DOMContentLoaded', () => {
    const trainBtn = document.getElementById('train_ai_btn');
    if (trainBtn) {
        trainBtn.addEventListener('click', async () => {
            const statusDiv = document.getElementById('train_status');
            const statusText = document.getElementById('train_status_text');
            
            if (statusDiv && statusText) {
                statusDiv.style.display = 'block';
                statusText.innerHTML = '🔄 Обучение началось, пожалуйста, подождите...';
                statusText.style.color = '#fbbf24';
            }
            
            await fetch('/ai/train', { method: 'POST' });
            
            const interval = setInterval(async () => {
                const statusResp = await fetch('/ai/status');
                const status = await statusResp.json();
                
                if (status.status === 'completed') {
                    clearInterval(interval);
                    if (statusText) {
                        statusText.innerHTML = status.message;
                        statusText.style.color = '#4ade80';
                    }
                } else if (status.status === 'failed') {
                    clearInterval(interval);
                    if (statusText) {
                        statusText.innerHTML = status.message;
                        statusText.style.color = '#f87171';
                    }
                } else if (status.status === 'in_progress' && statusText) {
                    statusText.innerHTML = status.message;
                }
            }, 2000);
        });
    }
    
    // --- ИИ предсказание (ПРЯМАЯ ПРИВЯЗКА) ---
    const predictBtn = document.getElementById('predict_ai_btn');
    if (predictBtn) {
        predictBtn.onclick = async () => {
            const h = document.getElementById('height_ai').value;
            try {
                const response = await fetch(`/ai/predict?height=${h}`);
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('ai_T').innerText = data.temperature_K;
                    document.getElementById('ai_P').innerText = data.pressure_Pa;
                    document.getElementById('ai_rho').innerText = data.density_kg_m3;
                    document.getElementById('ai_a').innerText = data.speed_of_sound_m_s;
                } else {
                    alert('Ошибка: модель не обучена. Сначала обучите нейросеть.');
                }
            } catch (error) {
                console.error('Ошибка предсказания:', error);
                alert('Ошибка соединения с сервером');
            }
        };
    }
});

// --- Верификация ---
const verifyBtn = document.getElementById('verify_btn');
if (verifyBtn) {
    verifyBtn.addEventListener('click', async () => {
        const res = await fetch('/verify/');
        if (!res.ok) { alert('Сначала обучите модель'); return; }
        const metrics = await res.json();
        let html = '<table><tr><th>Параметр</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>';
        metrics.forEach(m => {
            html += `<tr><td>${m.parameter}</td><td>${m.mae.toFixed(4)}</td><td>${m.rmse.toFixed(4)}</td><td>${m.r2.toFixed(4)}</td></tr>`;
        });
        html += '</table>';
        document.getElementById('verify_table').innerHTML = html;
    });
}

// --- Экспорт (только CSV и JSON) ---
const exportCsv = document.getElementById('export_csv');
if (exportCsv) exportCsv.addEventListener('click', () => window.open('/export/full_profile?format=csv', '_blank'));
const exportJson = document.getElementById('export_json');
if (exportJson) exportJson.addEventListener('click', () => window.open('/export/full_profile?format=json', '_blank'));

const exportTxt = document.getElementById('export_txt');
if (exportTxt) {
    exportTxt.addEventListener('click', () => window.open('/export/full_profile?format=txt', '_blank'));
}

const exportXml = document.getElementById('export_xml');
if (exportXml) {
    exportXml.addEventListener('click', () => window.open('/export/full_profile?format=xml', '_blank'));
}