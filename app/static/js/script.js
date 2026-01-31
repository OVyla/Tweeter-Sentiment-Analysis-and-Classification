const translations = {
    en: {
        title: "Analyze Tweet Sentiment<br>with AI",
        subtitle: "Input any text and let our advanced machine learning models determine if the sentiment is positive, neutral, or negative.",
        placeholder: "Type or paste a tweet here... (e.g. 'This movie was absolutely fantastic!')",
        analyzeBtn: "Analyze Sentiment",
        modelTitle: "The Model",
        modelDesc: "Powered by Logistic Regression with TF-IDF vectorization, trained on over 1.5 million tweets for high-speed, accurate classification.",
        techTitle: "Tech Stack",
        techDesc: "Built with Python, Scikit-learn, and FastAPI. Designed for performance and scalability.",
        negative: "Negative",
        neutral: "Neutral",
        positive: "Positive",
        howTitle: "How It Works",
        step1Title: "Pre-processing",
        step1Desc: "Cleaning tweets by removing noise, handling emojis, and lemmatizing text using NLTK.",
        step2Title: "Vectorization",
        step2Desc: "Converting text into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).",
        step3Title: "Classification",
        step3Desc: "Our Logistic Regression model predicts the sentiment probability based on learned patterns.",
        perfTitle: "Model Performance",
        perfSubtitle: "Benchmarking against other models on 1.5M tweets",
        accChartTitle: "Accuracy Comparison",
        f1ChartTitle: "F1-Score Comparison"
    },
    ca: {
        title: "Analitza el Sentiment de Tuits<br>amb IA",
        subtitle: "Introdueix qualsevol text i deixa que els nostres models d'aprenentatge automàtic determinin si el sentiment és positiu, neutre o negatiu.",
        placeholder: "Escriu o enganxa un tuit aquí... (ex. 'Aquesta pel·lícula ha estat absolutament fantàstica!')",
        analyzeBtn: "Analitza el Sentiment",
        modelTitle: "El Model",
        modelDesc: "Impulsat per Regressió Logística amb vectorització TF-IDF, entrenat amb més d'1.5 milions de tuits per a una classificació ràpida i precisa.",
        techTitle: "Tecnologia",
        techDesc: "Construït amb Python, Scikit-learn i FastAPI. Dissenyat per a rendiment i escalabilitat.",
        negative: "Negatiu",
        neutral: "Neutre",
        positive: "Positiu",
        howTitle: "Com Funciona",
        step1Title: "Pre-processament",
        step1Desc: "Neteja de tuits eliminant soroll, emojis i lematitzant el text amb NLTK.",
        step2Title: "Vectorització",
        step2Desc: "Conversió de text en vectors numèrics mitjançant TF-IDF (Freqüència de Terme - Freqüència Inversa de Document).",
        step3Title: "Classificació",
        step3Desc: "El nostre model de Regressió Logística prediu la probabilitat de sentiment basant-se en patrons apresos.",
        perfTitle: "Rendiment del Model",
        perfSubtitle: "Comparativa amb altres models en 1.5M de tuits",
        accChartTitle: "Comparació d'Accuracy",
        f1ChartTitle: "Comparació de F1-Score"
    }
};

let currentLang = 'en';

document.addEventListener('DOMContentLoaded', () => {
    // Language Switcher
    const btnEn = document.getElementById('btn-en');
    const btnCa = document.getElementById('btn-ca');

    btnEn.addEventListener('click', () => setLanguage('en'));
    btnCa.addEventListener('click', () => setLanguage('ca'));

    // Analyze Button
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.addEventListener('click', analyzeSentiment);

    // Render Charts
    renderCharts();
});

function renderCharts() {
    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    const ctxF1 = document.getElementById('f1Chart').getContext('2d');

    const models = ['Logistic Regression (TF-IDF)', 'SVM (TF-IDF)', 'Random Forest', 'Decision Tree', 'Naive Bayes'];
    const accData = [0.802, 0.799, 0.674, 0.566, 0.540];
    const f1Data = [0.803, 0.799, 0.675, 0.560, 0.538];
    const colors = [
        'rgba(29, 155, 240, 0.8)', // Primary
        'rgba(168, 85, 247, 0.7)', // Purple
        'rgba(16, 185, 129, 0.7)', // Green
        'rgba(245, 158, 11, 0.7)', // Orange
        'rgba(239, 68, 68, 0.7)'   // Red
    ];
    const borders = [
        'rgba(29, 155, 240, 1)',
        'rgba(168, 85, 247, 1)',
        'rgba(16, 185, 129, 1)',
        'rgba(245, 158, 11, 1)',
        'rgba(239, 68, 68, 1)'
    ];

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(0,0,0,0.8)',
                titleColor: '#fff',
                bodyColor: '#fff',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1.0,
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#8899a6' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#8899a6', font: { size: 10 } }
            }
        }
    };

    new Chart(ctxAcc, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'Accuracy',
                data: accData,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: chartOptions
    });

    new Chart(ctxF1, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'F1 Score',
                data: f1Data,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: chartOptions
    });
}

function setLanguage(lang) {
    currentLang = lang;
    const t = translations[lang];

    document.getElementById('btn-en').classList.toggle('active', lang === 'en');
    document.getElementById('btn-ca').classList.toggle('active', lang === 'ca');

    document.getElementById('title').innerHTML = t.title;
    document.getElementById('subtitle').innerText = t.subtitle;
    document.getElementById('tweet-input').placeholder = t.placeholder;
    document.getElementById('analyze-text').innerText = t.analyzeBtn;
    document.getElementById('model-title').innerText = t.modelTitle;
    document.getElementById('model-desc').innerText = t.modelDesc;
    document.getElementById('tech-title').innerText = t.techTitle;
    document.getElementById('tech-desc').innerText = t.techDesc;

    // New sections
    document.getElementById('how-title').innerText = t.howTitle;
    document.getElementById('step1-title').innerText = t.step1Title;
    document.getElementById('step1-desc').innerText = t.step1Desc;
    document.getElementById('step2-title').innerText = t.step2Title;
    document.getElementById('step2-desc').innerText = t.step2Desc;
    document.getElementById('step3-title').innerText = t.step3Title;
    document.getElementById('step3-desc').innerText = t.step3Desc;
    document.getElementById('perf-title').innerText = t.perfTitle;
    document.getElementById('perf-subtitle').innerText = t.perfSubtitle;
    document.getElementById('acc-chart-title').innerText = t.accChartTitle;
    document.getElementById('f1-chart-title').innerText = t.f1ChartTitle;
}

async function analyzeSentiment() {
    const input = document.getElementById('tweet-input').value;
    if (!input.trim()) return;

    const btn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');

    // UI Loading state
    btn.disabled = true;
    btn.style.opacity = '0.7';
    loader.classList.remove('hidden');
    resultContainer.classList.add('hidden');

    try {
        const formData = new FormData(); // Or JSON

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: input })
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data.sentiment);
        } else {
            alert('Error analyzing sentiment: ' + data.error);
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Something went wrong!');
    } finally {
        btn.disabled = false;
        btn.style.opacity = '1';
        loader.classList.add('hidden');
    }
}

function displayResult(sentiment) {
    const resultContainer = document.getElementById('result-container');
    const label = document.getElementById('sentiment-label');
    const icon = document.getElementById('sentiment-icon');
    const fill = document.getElementById('score-fill');

    resultContainer.classList.remove('hidden', 'positive', 'negative', 'neutral');

    let iconClass = '';
    let text = '';

    // Normalize sentiment string
    const sentimentLower = sentiment.toLowerCase();

    if (sentimentLower.includes('positive')) {
        resultContainer.classList.add('positive');
        iconClass = 'fa-regular fa-face-smile';
        text = translations[currentLang].positive;
        fill.style.width = '100%';
    } else if (sentimentLower.includes('negative')) {
        resultContainer.classList.add('negative');
        iconClass = 'fa-regular fa-face-frown';
        text = translations[currentLang].negative;
        fill.style.width = '100%';
    } else {
        resultContainer.classList.add('neutral');
        iconClass = 'fa-regular fa-face-meh';
        text = translations[currentLang].neutral;
        fill.style.width = '50%'; // Or calculate confidence if available
    }

    icon.innerHTML = `<i class="${iconClass}"></i>`;
    label.innerText = text;
}
