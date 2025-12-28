import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:before_after/before_after.dart';

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List? originalBytes, denoisedBytes;
  double psnr = 0, ssim = 0, mse = 0;
  double denoiseStrength = 1.0;
  bool sharpen = false;
  final picker = ImagePicker();

  // PICK IMAGE
  Future pickImage() async {
    final img = await picker.pickImage(source: ImageSource.gallery);
    if (img != null) {
      originalBytes = await img.readAsBytes();
      denoisedBytes = null;
      setState(() {});
    }
  }

  // CALL BACKEND API
  Future denoise() async {
    if (originalBytes == null) return;

    final url = Uri.parse("http://127.0.0.1:5000/denoise");

    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "image": base64Encode(originalBytes!),
        "strength": denoiseStrength,
        "sharpen": sharpen
      }),
    );

    final data = jsonDecode(response.body);

    setState(() {
      denoisedBytes = base64Decode(data['denoised_image']);
      psnr = data['metrics']['psnr'];
      ssim = data['metrics']['ssim'];
      mse  = data['metrics']['mse'];
    });
  }

  // SAVE OUTPUT (placeholder)
  void saveOutput() {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("💾 Save feature will be added soon...")),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xfff7eef7),
      appBar: AppBar(
        title: Text("Image Denoising Autoencoder"),
        backgroundColor: Colors.deepPurple,
      ),

      body: Column(
        children: [

          // SHOW ORIGINAL BEFORE DENOISE
          if (originalBytes != null && denoisedBytes == null)
            Expanded(
              child: Center(child: Image.memory(originalBytes!)),
            ),

          // SHOW BEFORE / AFTER SLIDER
          if (originalBytes != null && denoisedBytes != null)
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: BeforeAfter(
                  before: Image.memory(originalBytes!),
                  after: Image.memory(denoisedBytes!),
                ),
              ),
            ),

          // 📌 TECHNICAL METRICS DISPLAY
          if (denoisedBytes != null) ...[
            SizedBox(height: 10),
            Container(
              width: double.infinity,
              margin: EdgeInsets.symmetric(horizontal: 20),
              padding: EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(15),
                boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 6)],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("📊 Technical Parameters",
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                  SizedBox(height: 10),
                  Text("• PSNR : $psnr dB (Higher = Better)"),
                  Text("• SSIM : $ssim (Closer to 1 = More Similar)"),
                  Text("• MSE  : $mse (Lower = Less Noise)"),
                ],
              ),
            ),
          ],

          SizedBox(height: 10),

          // 🔧 STRENGTH SLIDER
          Text("Denoise Strength: ${(denoiseStrength * 100).round()}%"),
          Slider(
            value: denoiseStrength,
            min: 0.1,
            max: 1.0,
            divisions: 9,
            label: "${(denoiseStrength * 100).round()}%",
            onChanged: (v) => setState(() => denoiseStrength = v),
          ),

          // SHARPEN OPTION
          SwitchListTile(
            title: Text("Sharpen Output"),
            value: sharpen,
            onChanged: (v) => setState(() => sharpen = v),
          ),

          SizedBox(height: 10),

          // 👉 BUTTONS
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(onPressed: pickImage, child: Text("📁 Pick Image")),
              SizedBox(width: 10),
              ElevatedButton(onPressed: denoise, child: Text("✨ Denoise")),
              SizedBox(width: 10),
              ElevatedButton(onPressed: saveOutput, child: Text("💾 Save")),
            ],
          ),

          SizedBox(height: 15),
        ],
      ),
    );
  }
}
