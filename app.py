from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = "goagro_secret_2025"

# Load trained model
model = joblib.load("crop_model.pkl")

# ── Full crop knowledge base ──────────────────────────────────────────────────
CROP_DATA = {
    "rice": {
        "image": "https://images.unsplash.com/photo-1586201375761-83865001e31c?w=800",
        "banner": "https://images.unsplash.com/photo-1536054945-2f4e5e5e5e5e?w=1200",
        "season": "Kharif (June – November)",
        "duration": "90 – 150 days",
        "water": "High (1200–2000 mm)",
        "temp_range": "20 – 35°C",
        "ph_range": "5.0 – 7.0",
        "soil_type": "Clay / Loamy",
        "market_price": "₹1,940 / quintal (MSP)",
        "yield": "2 – 4 tonnes/ha",
        "description": "Rice is the staple food for more than half the world's population. It thrives in warm, humid conditions with standing water. India is the second-largest producer globally.",
        "tips": [
            "Maintain 5–10 cm standing water during vegetative stage",
            "Apply nitrogen in 3 splits — basal, tillering, panicle initiation",
            "Use SRI (System of Rice Intensification) to boost yield by 20–30%",
            "Watch for blast disease in humid conditions — apply tricyclazole",
            "Drain field 10 days before harvest for easy mechanized cutting"
        ],
        "fertilizer": {"N": "120 kg/ha", "P": "60 kg/ha", "K": "60 kg/ha"},
        "diseases": ["Rice Blast", "Brown Plant Hopper", "Sheath Blight", "Bacterial Leaf Blight"],
        "color": "#4ade80"
    },
    "wheat": {
        "image": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=800",
        "banner": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=1200",
        "season": "Rabi (October – March)",
        "duration": "100 – 150 days",
        "water": "Moderate (450–650 mm)",
        "temp_range": "10 – 25°C",
        "ph_range": "6.0 – 7.5",
        "soil_type": "Well-drained Loamy",
        "market_price": "₹2,275 / quintal (MSP)",
        "yield": "3 – 5 tonnes/ha",
        "description": "Wheat is the most widely cultivated cereal crop globally. It is a cool-season crop that requires well-drained fertile soils. India is the second-largest producer after China.",
        "tips": [
            "Sow between Oct 15 – Nov 15 for optimal yield in North India",
            "First irrigation (Crown Root Initiation) is critical at 20–25 DAS",
            "Apply zinc sulfate if soil Zn is below 0.6 ppm",
            "Use resistant varieties like HD-2967 to combat rust diseases",
            "Harvest when grain moisture drops to 12–14%"
        ],
        "fertilizer": {"N": "120 kg/ha", "P": "60 kg/ha", "K": "40 kg/ha"},
        "diseases": ["Yellow Rust", "Brown Rust", "Loose Smut", "Karnal Bunt"],
        "color": "#f59e0b"
    },
    "maize": {
        "image": "https://images.unsplash.com/photo-1601597111158-2fceff292cdc?w=800",
        "banner": "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=1200",
        "season": "Kharif & Rabi",
        "duration": "80 – 110 days",
        "water": "Moderate (500–800 mm)",
        "temp_range": "18 – 32°C",
        "ph_range": "5.8 – 7.0",
        "soil_type": "Sandy Loam / Loamy",
        "market_price": "₹1,870 / quintal (MSP)",
        "yield": "4 – 8 tonnes/ha",
        "description": "Maize is a versatile crop used for food, feed, and industrial purposes. It is the third most important cereal crop in India and has the highest genetic yield potential among cereals.",
        "tips": [
            "Maintain plant population of 65,000–75,000 plants/ha",
            "Apply 25% N as basal and 75% in 2 top-dressings",
            "Earthing up at 30–35 DAS prevents lodging",
            "Irrigate at knee-high, tasseling, and grain-filling stages",
            "Harvest when husk turns brown and grain moisture is 25–30%"
        ],
        "fertilizer": {"N": "150 kg/ha", "P": "75 kg/ha", "K": "75 kg/ha"},
        "diseases": ["Fall Armyworm", "Turcicum Blight", "Downy Mildew", "Stalk Rot"],
        "color": "#f97316"
    },
    "chickpea": {
        "image": "https://images.unsplash.com/photo-1515543904379-3d757afe72e4?w=800",
        "banner": "https://images.unsplash.com/photo-1515543904379-3d757afe72e4?w=1200",
        "season": "Rabi (October – March)",
        "duration": "90 – 120 days",
        "water": "Low (300–400 mm)",
        "temp_range": "15 – 29°C",
        "ph_range": "6.0 – 8.0",
        "soil_type": "Sandy Loam / Medium Black",
        "market_price": "₹5,440 / quintal (MSP)",
        "yield": "1 – 2 tonnes/ha",
        "description": "Chickpea (Chana) is the most important pulse crop in India. It fixes atmospheric nitrogen, improving soil fertility. India accounts for 70% of global chickpea production.",
        "tips": [
            "Seed treatment with Rhizobium culture improves nitrogen fixation",
            "Avoid waterlogging — chickpea is highly sensitive to excess moisture",
            "One pre-sowing irrigation is sufficient in most cases",
            "Spray oxalic acid to reduce pod borer damage",
            "Harvest when 70–80% pods turn brown"
        ],
        "fertilizer": {"N": "20 kg/ha", "P": "60 kg/ha", "K": "20 kg/ha"},
        "diseases": ["Fusarium Wilt", "Ascochyta Blight", "Pod Borer", "Collar Rot"],
        "color": "#eab308"
    },
    "cotton": {
        "image": "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?w=800",
        "banner": "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?w=1200",
        "season": "Kharif (April – November)",
        "duration": "150 – 180 days",
        "water": "Moderate (700–1200 mm)",
        "temp_range": "21 – 35°C",
        "ph_range": "6.0 – 8.0",
        "soil_type": "Black Cotton Soil / Deep Loam",
        "market_price": "₹6,620 / quintal (MSP)",
        "yield": "1.5 – 3 tonnes/ha",
        "description": "Cotton is the most important fiber crop and a major cash crop in India. Bt cotton has revolutionized production by controlling bollworm. India is the largest producer of cotton globally.",
        "tips": [
            "Use Bt cotton hybrids for bollworm resistance",
            "Maintain optimum plant population of 11,000–16,000 plants/ha",
            "Apply boron and zinc micronutrients for better boll development",
            "Monitor for whitefly and pink bollworm regularly",
            "Pick cotton in dry weather to maintain fiber quality"
        ],
        "fertilizer": {"N": "120 kg/ha", "P": "60 kg/ha", "K": "60 kg/ha"},
        "diseases": ["Bollworm", "Whitefly", "Fusarium Wilt", "Leaf Curl Virus"],
        "color": "#a78bfa"
    },
    "sugarcane": {
        "image": "https://images.unsplash.com/photo-1596040033229-a9821ebd058d?w=800",
        "banner": "https://images.unsplash.com/photo-1596040033229-a9821ebd058d?w=1200",
        "season": "Year-round (planted Feb–Mar)",
        "duration": "10 – 18 months",
        "water": "High (1500–2500 mm)",
        "temp_range": "20 – 35°C",
        "ph_range": "6.0 – 7.5",
        "soil_type": "Deep Loamy / Clay Loam",
        "market_price": "₹315 / quintal (FRP)",
        "yield": "60 – 100 tonnes/ha",
        "description": "Sugarcane is the primary source of sugar and ethanol in India. It is a long-duration crop that requires high water and nutrient inputs. India is the second-largest producer after Brazil.",
        "tips": [
            "Use disease-free setts from certified seed cane",
            "Trash mulching conserves moisture and suppresses weeds",
            "Apply ratoon management for 2–3 ratoon crops",
            "Intercrop with legumes in early stages to maximize land use",
            "Harvest at 10–12 months for optimal sucrose content"
        ],
        "fertilizer": {"N": "250 kg/ha", "P": "115 kg/ha", "K": "115 kg/ha"},
        "diseases": ["Red Rot", "Smut", "Wilt", "Pyrilla (Leafhopper)"],
        "color": "#34d399"
    },
    "mango": {
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?w=800",
        "banner": "https://images.unsplash.com/photo-1553279768-865429fa0078?w=1200",
        "season": "Summer (March – June)",
        "duration": "3–5 years to first fruit",
        "water": "Moderate (750–2500 mm)",
        "temp_range": "24 – 35°C",
        "ph_range": "5.5 – 7.5",
        "soil_type": "Deep Well-drained Loamy",
        "market_price": "₹30–80 / kg",
        "yield": "10 – 20 tonnes/ha",
        "description": "Mango is the national fruit of India and the king of fruits. India produces 40% of the world's mangoes. Alphonso, Dasheri, and Langra are premium varieties with high export demand.",
        "tips": [
            "Plant at 10×10 m spacing for standard varieties",
            "Paclobutrazol application induces flowering in off-season",
            "Bagging of fruits prevents fruit fly damage",
            "Apply potassium-rich fertilizer before flowering",
            "Harvest at physiological maturity — not full ripeness"
        ],
        "fertilizer": {"N": "1 kg/tree/year", "P": "0.5 kg/tree/year", "K": "1 kg/tree/year"},
        "diseases": ["Anthracnose", "Powdery Mildew", "Mango Hopper", "Fruit Fly"],
        "color": "#fb923c"
    },
    "banana": {
        "image": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=800",
        "banner": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=1200",
        "season": "Year-round",
        "duration": "11 – 15 months",
        "water": "High (1200–2200 mm)",
        "temp_range": "15 – 35°C",
        "ph_range": "6.0 – 7.5",
        "soil_type": "Rich Loamy / Alluvial",
        "market_price": "₹15–25 / kg",
        "yield": "30 – 50 tonnes/ha",
        "description": "Banana is the most widely consumed fruit in India and a major commercial crop. It is a high-value crop with year-round production. Tissue culture plants give uniform and disease-free crop.",
        "tips": [
            "Use tissue culture plants for disease-free uniform crop",
            "Propping is essential to prevent toppling at bunch stage",
            "Bunch covering with blue polythene bags improves quality",
            "Fertigation through drip irrigation saves 30–40% water",
            "Desuckering — keep only one follower sucker per plant"
        ],
        "fertilizer": {"N": "200 kg/ha", "P": "60 kg/ha", "K": "300 kg/ha"},
        "diseases": ["Panama Wilt", "Sigatoka Leaf Spot", "Bunchy Top Virus", "Nematodes"],
        "color": "#fbbf24"
    },
    "grapes": {
        "image": "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=800",
        "banner": "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=1200",
        "season": "Rabi (Oct–Feb harvest)",
        "duration": "Perennial (3+ years)",
        "water": "Moderate (700–900 mm)",
        "temp_range": "15 – 40°C",
        "ph_range": "6.5 – 7.5",
        "soil_type": "Sandy Loam / Gravelly",
        "market_price": "₹40–120 / kg",
        "yield": "15 – 25 tonnes/ha",
        "description": "Grapes are a high-value horticultural crop grown mainly in Maharashtra, Karnataka, and Andhra Pradesh. India exports table grapes to Europe and the Middle East.",
        "tips": [
            "Pruning is the most critical operation — determines yield",
            "Gibberellic acid (GA3) spray improves berry size",
            "Drip irrigation with fertigation is essential",
            "Downy mildew control requires preventive fungicide sprays",
            "Harvest at 18–22° Brix for export quality"
        ],
        "fertilizer": {"N": "150 kg/ha", "P": "75 kg/ha", "K": "150 kg/ha"},
        "diseases": ["Downy Mildew", "Powdery Mildew", "Anthracnose", "Botrytis"],
        "color": "#c084fc"
    },
    "lentil": {
        "image": "https://images.unsplash.com/photo-1515543904379-3d757afe72e4?w=800",
        "banner": "https://images.unsplash.com/photo-1515543904379-3d757afe72e4?w=1200",
        "season": "Rabi (October – April)",
        "duration": "100 – 130 days",
        "water": "Low (250–400 mm)",
        "temp_range": "18 – 30°C",
        "ph_range": "6.0 – 8.0",
        "soil_type": "Sandy Loam / Loamy",
        "market_price": "₹6,000 / quintal (MSP)",
        "yield": "0.8 – 1.5 tonnes/ha",
        "description": "Lentil (Masoor) is an important pulse crop rich in protein. It is a drought-tolerant crop that improves soil fertility through nitrogen fixation. India is a major producer and consumer.",
        "tips": [
            "Seed inoculation with Rhizobium is highly beneficial",
            "Avoid heavy soils — lentil is sensitive to waterlogging",
            "One pre-sowing irrigation is usually sufficient",
            "Weed management in first 30–40 days is critical",
            "Harvest when 70% pods turn yellow-brown"
        ],
        "fertilizer": {"N": "20 kg/ha", "P": "40 kg/ha", "K": "20 kg/ha"},
        "diseases": ["Rust", "Wilt", "Stemphylium Blight", "Aphids"],
        "color": "#f87171"
    },
    "pomegranate": {
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=800",
        "banner": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=1200",
        "season": "Mrig bahar (June–July)",
        "duration": "5–6 months per crop",
        "water": "Low–Moderate (500–800 mm)",
        "temp_range": "25 – 38°C",
        "ph_range": "6.5 – 7.5",
        "soil_type": "Well-drained Sandy Loam",
        "market_price": "₹80–150 / kg",
        "yield": "15 – 25 tonnes/ha",
        "description": "Pomegranate is a drought-tolerant, high-value fruit crop grown in arid and semi-arid regions. Maharashtra's Bhagwa variety is world-renowned for its quality and export potential.",
        "tips": [
            "Bahar treatment (stress + fertilizer) controls flowering time",
            "Fruit cracking is reduced by uniform irrigation",
            "Bag fruits to prevent fruit borer and improve color",
            "Bacterial blight is the most serious disease — use copper sprays",
            "Harvest at 135–145 days after fruit set"
        ],
        "fertilizer": {"N": "625 g/tree", "P": "250 g/tree", "K": "200 g/tree"},
        "diseases": ["Bacterial Blight", "Fruit Borer", "Cercospora Leaf Spot", "Aphids"],
        "color": "#fb7185"
    },
    "watermelon": {
        "image": "https://images.unsplash.com/photo-1563114773-84221bd62daa?w=800",
        "banner": "https://images.unsplash.com/photo-1563114773-84221bd62daa?w=1200",
        "season": "Summer (Feb – May)",
        "duration": "70 – 90 days",
        "water": "Moderate (400–600 mm)",
        "temp_range": "24 – 35°C",
        "ph_range": "6.0 – 7.0",
        "soil_type": "Sandy Loam",
        "market_price": "₹8–15 / kg",
        "yield": "25 – 40 tonnes/ha",
        "description": "Watermelon is a warm-season cucurbit crop with high water content and market demand in summer. It is a short-duration, high-return crop suitable for sandy soils along river banks.",
        "tips": [
            "Raised bed cultivation with drip irrigation is ideal",
            "Pollination by bees is essential — avoid insecticides during flowering",
            "Fruit thumping test: hollow sound indicates ripeness",
            "Mulching with black polythene controls weeds and conserves moisture",
            "Harvest when tendril near fruit dries up"
        ],
        "fertilizer": {"N": "100 kg/ha", "P": "50 kg/ha", "K": "75 kg/ha"},
        "diseases": ["Downy Mildew", "Fusarium Wilt", "Anthracnose", "Fruit Fly"],
        "color": "#4ade80"
    },
}

# Default fallback for crops not in the knowledge base
DEFAULT_CROP = {
    "image": "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=800",
    "banner": "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=1200",
    "season": "Varies by region",
    "duration": "Varies",
    "water": "Moderate",
    "temp_range": "15 – 35°C",
    "ph_range": "6.0 – 7.5",
    "soil_type": "Loamy",
    "market_price": "Market dependent",
    "yield": "Varies",
    "description": "This crop is well-suited to your current soil and climate conditions based on our ML analysis.",
    "tips": [
        "Test soil regularly and amend based on results",
        "Use certified seeds from reputable sources",
        "Follow integrated pest management practices",
        "Maintain proper irrigation schedule",
        "Consult local agricultural extension officer for region-specific advice"
    ],
    "fertilizer": {"N": "As per soil test", "P": "As per soil test", "K": "As per soil test"},
    "diseases": ["Monitor regularly for local pest and disease pressure"],
    "color": "#4ade80"
}


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/predict-page')
def predict_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        N        = float(request.form['N'])
        P        = float(request.form['P'])
        K        = float(request.form['K'])
        temp     = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph       = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Basic validation
        if not (0 <= ph <= 14):
            raise ValueError("pH out of range")
        if not (0 <= humidity <= 100):
            raise ValueError("Humidity out of range")

        data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        prediction = model.predict(data)[0].lower()

        # Store inputs in session for result page
        session['result']      = prediction
        session['N']           = N
        session['P']           = P
        session['K']           = K
        session['temperature'] = temp
        session['humidity']    = humidity
        session['ph']          = ph
        session['rainfall']    = rainfall

        return redirect(url_for('result'))

    except Exception as e:
        return render_template("index.html",
                               error="Please enter valid numeric values in all fields.")


@app.route('/result')
def result():
    if 'result' not in session:
        return redirect(url_for('predict_page'))

    crop_name = session['result']
    info      = CROP_DATA.get(crop_name, DEFAULT_CROP)
    info['name'] = crop_name.capitalize()

    return render_template("result.html",
                           crop=info,
                           N=session['N'],
                           P=session['P'],
                           K=session['K'],
                           temperature=session['temperature'],
                           humidity=session['humidity'],
                           ph=session['ph'],
                           rainfall=session['rainfall'])


if __name__ == "__main__":
    app.run(debug=True)
