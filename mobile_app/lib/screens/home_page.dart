import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class HomePage extends StatefulWidget {
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List? originalImg;
  Uint8List? denoisedImg;
  bool loading = false;

  double strength = 1.0;

  // 🔥 Sharpen ON by default (no toggle)
  final bool sharpen = true;

  double? psnr, ssim, mse;

  final picker = ImagePicker();
  final String api = "http://127.0.0.1:5000/denoise"; // Backend URL

  Future<void> pickImage() async {
    final file = await picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;

    final bytes = await file.readAsBytes();
    setState(() {
      originalImg = bytes;
      denoisedImg = null;
      psnr = ssim = mse = null;
    });
  }

  Future<void> runDenoise() async {
    if (originalImg == null) return;

    setState(() => loading = true);

    try {
      final body = jsonEncode({
        "image": base64Encode(originalImg!),
        "strength": strength,
        "sharpen": true // 👈 ALWAYS TRUE
      });

      final res = await http.post(
        Uri.parse(api),
        headers: {"Content-Type": "application/json"},
        body: body,
      );

      if (res.statusCode != 200) {
        setState(() => loading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("❌ Backend error: ${res.body}")),
        );
        return;
      }

      final data = jsonDecode(res.body);
      setState(() {
        denoisedImg = base64Decode(data["image"]);
        psnr = data["psnr"];
        ssim = data["ssim"];
        mse = data["mse"];
        loading = false;
      });
    } catch (e) {
      setState(() => loading = false);
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text("⚠ Error: $e")));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("🧽 Image Denoising App"), centerTitle: true),

      body: Padding(
        padding: EdgeInsets.all(12),
        child: Column(
          children: [

            Expanded(
              child: Container(
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.blueAccent),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: originalImg == null
                    ? Text("📸 Select an image to start")
                    : denoisedImg == null
                        ? Image.memory(originalImg!)
                        : Row(
                            children: [
                              Expanded(child: Image.memory(originalImg!, fit: BoxFit.contain)),
                              Container(width: 2, color: Colors.white24),
                              Expanded(child: Image.memory(denoisedImg!, fit: BoxFit.contain)),
                            ],
                          ),
              ),
            ),

            SizedBox(height: 12),

            Text("Denoise Strength", style: TextStyle(fontSize: 16)),
            Slider(
              value: strength,
              min: 0.0,
              max: 1.0,
              divisions: 10,
              label: strength.toStringAsFixed(1),
              onChanged: (v) => setState(() => strength = v),
            ),

            SizedBox(height: 10),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: pickImage,
                    child: Text("📁 Pick Image"),
                  ),
                ),
                SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton(
                    onPressed: loading ? null : runDenoise,
                    child: loading
                        ? CircularProgressIndicator()
                        : Text("🚀 Denoise"),
                  ),
                ),
              ],
            ),

            SizedBox(height: 10),

            if (psnr != null)
              Container(
                padding: EdgeInsets.all(10),
                decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(10)),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("📊 Metrics:", style: TextStyle(fontWeight: FontWeight.bold)),
                    Text("• PSNR  : ${psnr!.toStringAsFixed(2)}"),
                    Text("• SSIM  : ${ssim!.toStringAsFixed(4)}"),
                    Text("• MSE   : ${mse!.toStringAsFixed(2)}"),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
