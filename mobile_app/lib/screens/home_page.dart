import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List? originalImg;
  Uint8List? denoisedImg;
  bool loading = false;

  double strength = 1.0;
  bool sharpen = false;

  double? psnr, ssim, mse;

  final picker = ImagePicker();
  final String api = "http://127.0.0.1:5000/denoise"; // 👈 Backend Address

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
        "sharpen": sharpen
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
      appBar: AppBar(title: const Text("🧽 Image Denoising App"), centerTitle: true),

      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            // 📌 Image Preview Area
            Expanded(
              child: Container(
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.blueAccent),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: originalImg == null
                    ? const Text("📸 Select an image to start")
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

            const SizedBox(height: 12),

            // 📌 Strength Slider
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text("Denoise Strength", style: TextStyle(fontSize: 16)),
                Slider(
                  value: strength,
                  min: 0.0,
                  max: 1.0,
                  divisions: 10,
                  label: strength.toStringAsFixed(1),
                  onChanged: (v) => setState(() => strength = v),
                ),
              ],
            ),

            // 📌 Sharpen Toggle
            SwitchListTile(
              value: sharpen,
              title: const Text("Sharpen Output"),
              onChanged: (v) => setState(() => sharpen = v),
            ),

            const SizedBox(height: 8),

            // 📌 Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: pickImage,
                    child: const Text("📁 Pick Image"),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton(
                    onPressed: loading ? null : runDenoise,
                    child: loading
                        ? const CircularProgressIndicator()
                        : const Text("🚀 Denoise"),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 10),

            // 📌 Metrics Display
            if (psnr != null)
              Container(
                padding: const EdgeInsets.all(10),
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
