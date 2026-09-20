from flask import Flask, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
from PIL import Image
import requests
import io
import os
import json

app = Flask(__name__)

# 1. ดึง Gemini API Key จาก Environment Variable บน Render
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash-lite')

# 2. ดึงข้อมูล Google Sheets Credentials (JSON) จาก Environment Variable
google_creds_json = os.environ.get("GOOGLE_CREDS_JSON")
creds_data = json.loads(google_creds_json)

# จัดการแปลงอักขระ \n ใน private_key ป้องกันปัญหา Invalid JWT Signature บน Cloud
if "private_key" in creds_data:
    creds_data["private_key"] = creds_data["private_key"].replace("\\n", "\n")

scope = ["https://www.spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(creds_data, scopes=scope)
client = gspread.authorize(creds)

# เปิดไฟล์ Google Sheets (ระบุชื่อไฟล์ชีทของคุณตรงนี้)
sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1

@app.route('/webhook', methods=['POST'])
def receive_image():
    try:
        # รับข้อมูล JSON ที่ส่งเข้ามา (คาดหวัง key ชื่อ image_url)
        data = request.json
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({"status": "error", "message": "ไม่พบลิงก์รูปภาพ"}), 400

        # ดาวน์โหลดรูปภาพจากลิงก์
        img_bytes = requests.get(image_url).content
        img = Image.open(io.BytesIO(img_bytes))
        
        # คำสั่ง Prompt ให้ Gemini สกัดข้อมูลตามโครงสร้าง 12 คอลัมน์
        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ,ค่าน้ำสูง,ค่าน้ำต่ำ,1X2 เหย้า,1X2 เสมอ,1X2 เยือน โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
        
        # ให้ Gemini ประมวลผลภาพ
        response = model.generate_content([img, prompt])
        raw_text = response.text.strip()
        
        # ตัดแบ่งข้อมูลด้วยลูกน้ำ (,)
        row_data = [item.strip() for item in raw_text.split(',')]
        
        # บันทึกลง Google Sheets ทันที
        sheet.append_row(row_data)
        
        return jsonify({"status": "success", "data": row_data}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
