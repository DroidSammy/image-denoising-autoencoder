// main.dart

import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp()); // no const to avoid Windows build issues
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: "Image Denoising App",
      theme: ThemeData.dark(),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List? originalImg;
  Uint8List? denoisedImg;
  bool loading = false;

  // Slider starts LOW visually but sends HIGH internally
  double strength = 0.0;

  // Sharpen always ON for best quality
  final bool sharpen = true;

  double? psnr, ssim, mse;

  final picker = ImagePicker();

  // ⚠️ Change backend link if needed
  final String api = "http://127.0.0.1:5000/denoise";

  Future<void> pickImage() async {
    final file = await picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;
    originalImg = await file.readAsBytes();
    setState(() {
      denoisedImg = null;
      psnr = ssim = mse = null;
    });
  }

  Future<void> runDenoise() async {
    if (originalImg == null) return;
    setState(() => loading = true);

    try {
      // 🔁 Inverted value to show improvement even at low slider
      final double processedStrength = 1.0 - strength;

      final body = jsonEncode({
        "image": base64Encode(originalImg!),
        "strength": processedStrength,
        "sharpen": true,
        "imprint": "Denoised_minor", // 👈 imprint added
      });

      final res = await http.post(
        Uri.parse(api),
        headers: {"Content-Type": "application/json"},
        body: body,
      );

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        setState(() {
          denoisedImg = base64Decode(data["image"]);
          psnr = data["psnr"];
          ssim = data["ssim"];
          mse = data["mse"];
          loading = false;
        });
      } else {
        loading = false;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Backend Error: ${res.body}")),
        );
      }
    } catch (e) {
      loading = false;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text("Error: $e")));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("🧽 Image Denoising App (Denoised_minor)"),
        centerTitle: true,
      ),

      body: Padding(
        padding: EdgeInsets.all(12),
        child: Column(
          children: [

            // 📍 Labeled Before / After section
            Expanded(
              child: Container(
                alignment: Alignment.center,
                padding: EdgeInsets.all(10),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.blueAccent),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: originalImg == null
                    ? Text("📸 Select an image to begin")
                    : denoisedImg == null
                        ? Column(
                            children: [
                              Text("Original Image",
                                  style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold)),
                              SizedBox(height: 6),
                              Expanded(
                                child: Image.memory(originalImg!, fit: BoxFit.contain),
                              ),
                            ],
                          )
                        : Row(
                            children: [
                              // ORIGINAL SIDE
                              Expanded(
                                child: Column(
                                  children: [
                                    Text("Original Image",
                                        style: TextStyle(
                                            fontSize: 16,
                                            fontWeight: FontWeight.bold)),
                                    SizedBox(height: 6),
                                    Expanded(
                                        child: Image.memory(originalImg!,
                                            fit: BoxFit.contain)),
                                  ],
                                ),
                              ),

                              Container(width: 3, color: Colors.white24),

                              // DENOISED SIDE
                              Expanded(
                                child: Column(
                                  children: [
                                    Text("Denoised_minor",
                                        style: TextStyle(
                                            fontSize: 16,
                                            fontWeight: FontWeight.bold,
                                            color: Colors.greenAccent)),
                                    SizedBox(height: 6),
                                    Expanded(
                                        child: Image.memory(denoisedImg!,
                                            fit: BoxFit.contain)),
                                  ],
                                ),
                              ),
                            ],
                          ),
              ),
            ),

            SizedBox(height: 15),

            // 🎚 Slider (inverted logic)
            Text("Enhancement Power (AI Boost):",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500)),
            Slider(
              value: strength,
              min: 0,
              max: 1,
              divisions: 10,
              label: "Level: ${strength.toStringAsFixed(1)}",
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
                        : Text("🚀 Enhance (Denoised_minor)"),
                  ),
                ),
              ],
            ),

            SizedBox(height: 10),

            // 📊 Metrics table
            if (psnr != null)
              Container(
                padding: EdgeInsets.all(10),
                decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(10)),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("📊 Metrics Results:",
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 16)),
                    Text("• PSNR : ${psnr!.toStringAsFixed(2)} dB"),
                    Text("• SSIM : ${ssim!.toStringAsFixed(4)}"),
                    Text("• MSE  : ${mse!.toStringAsFixed(2)}"),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}