#include <ESP8266WiFi.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// LCD I2C tại địa chỉ 0x27, kích thước 16x2
LiquidCrystal_I2C lcd(0x27, 16, 2);

// WiFi thông tin
const char* ssid = "Shine";
const char* password = "minhquyen";

WiFiServer server(80);

void setup() {
  Serial.begin(115200);

  Wire.begin(4, 5);  // D2 = SDA, D1 = SCL
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Connecting...");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("IP:");
  lcd.setCursor(0, 1);
  lcd.print(WiFi.localIP());

  server.begin();
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    Serial.println("Client connected");
    String req = "";

    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        req += c;

        // Nếu kết thúc header HTTP
        if (req.endsWith("\r\n\r\n")) {
          int idx = req.indexOf("/char?c=");
          if (idx != -1) {
            int start = idx + 8;
            int end = req.indexOf(' ', start);
            if (end == -1) end = req.length();
            String content = req.substring(start, end);

            content.replace("%20", " "); // Giải mã khoảng trắng

            Serial.print("Received string: ");
            Serial.println(content);

            // Hiển thị chuỗi lên LCD
            lcd.clear();
            lcd.setCursor(0, 0);
            lcd.print("Received:");
            lcd.setCursor(0, 1);
            lcd.print(content.substring(0, 16)); // Tối đa 16 ký tự
          }

          // Gửi phản hồi HTTP
          client.println("HTTP/1.1 200 OK");
          client.println("Content-Type: text/html");
          client.println("Connection: close");
          client.println();
          client.println("<html><body><h1>OK</h1></body></html>");
          break;
        }
      }
    }

    delay(1);
    client.stop();
    Serial.println("Client disconnected");
  }
}
