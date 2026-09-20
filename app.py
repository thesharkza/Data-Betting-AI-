from flask import Flask, request, jsonify
import gspread
import google.generativeai as genai
from PIL import Image
import requests
import io
import os

app = Flask(__name__)

# 1. ตั้งค่า Gemini API Key (ดึงจาก Environment Variable บน Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash-lite')

# 2. อ่านไฟล์ Credentials จาก Secret File ของ Render
# ตำแหน่งไฟล์ลับบน Render คือ /etc/secrets/credentials.json
creds_path = '/etc/secrets/credentials.json'

# ป้องกันกรณีรันทดสอบในคอมตัวเอง (ถ้าไม่พบใน Render ให้ใช้ไฟล์ในโฟลเดอร์เดียวกัน)
if not os.path.exists(creds_path):
    creds_path = 'credentials.json'

# 3. เชื่อมต่อ Google Sheets ด้วยวิธีที่อัปเดตล่าสุด (ระบบจัดการเรื่อง Token และ Scope ให้อัตโนมัติ)
client = gspread.service_account(filename=creds_path)

# เปิดไฟล์ Google Sheets (แก้ชื่อไฟล์ตรงนี้หากมีการเปลี่ยนชื่อ)
sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1

@app.route('/webhook', methods=['POST'])
def receive_image():
    try:
        # รับข้อมูล JSON จาก Webhook
        data = request.json
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({"status": "error", "message": "ไม่พบลิงก์รูปภาพ"}), 400

        # ดาวน์โหลดรูปภาพจากลิงก์
        img_bytes = requests.get(image_url).content
        img = Image.open(io.BytesIO(img_bytes))
        
        # คำสั่ง Prompt ให้ Gemini สกัดข้อมูล
        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ,ค่าน้ำสูง,ค่าน้ำต่ำ,1X2 เหย้า,1X2 เสมอ,1X2 เยือน โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
        
        # ส่งให้ Gemini ประมวลผล
        response = model.generate_content([img, prompt])
        raw_text = response.text.strip()
        
        # ตัดแบ่งข้อมูลด้วยลูกน้ำ (,) เพื่อเตรียมลงตาราง
        row_data = [item.strip() for item in raw_text.split(',')]
        
        # บันทึกลง Google Sheets บรรทัดถัดไปทันที
        sheet.append_row(row_data)
        
        return jsonify({"status": "success", "data": row_data}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
