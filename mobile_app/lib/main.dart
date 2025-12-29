import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() => runApp(MaterialApp(
  debugShowCheckedModeBanner: false,
  theme: ThemeData.dark(),
  home: HomePage(),
));

class HomePage extends StatefulWidget {
  @override State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List? originalImg, enhancedImg, diffImg;
  double strength = 0.10;
  bool loading = false;
  double? psnr, ssim, mse;

  final picker = ImagePicker();
  final String api = "http://127.0.0.1:5000/denoise";

  Future<void> pickImage() async {
    final file = await picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;
    originalImg = await file.readAsBytes();
    enhancedImg = diffImg = null;
    setState(() {});
  }

  Future<void> runDenoise() async {
    if (originalImg == null) return;
    setState(() => loading = true);

    final res = await http.post(
      Uri.parse(api),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "image": base64Encode(originalImg!),
        "strength": strength
      }),
    );

    final data = jsonDecode(res.body);
    enhancedImg = base64Decode(data["clean"]);
    diffImg = base64Decode(data["difference"]);
    psnr = data["psnr"]; ssim = data["ssim"]; mse = data["mse"];

    setState(() => loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("🔥 AI Denoising + Difference View")),

      body: Column(
        children: [
          Expanded(
            child: Row(
              children: [
                _panel("Original", originalImg),
                VerticalDivider(color: Colors.white),
                _panel("Enhanced", enhancedImg),
                VerticalDivider(color: Colors.white),
                _panel("Difference", diffImg), // 🔥 New heatmap
              ],
            ),
          ),

          Text("Strength: ${(strength*100).round()}%"),
          Slider(
            value: strength, min: 0, max: 1, divisions: 20,
            onChanged: (v) => setState(() => strength = v),
          ),

          Row(
            children: [
              Expanded(child: ElevatedButton(onPressed: pickImage, child: Text("📁 Pick Image"))),
              SizedBox(width: 10),
              Expanded(child: ElevatedButton(
                onPressed: loading ? null : runDenoise,
                child: loading ? CircularProgressIndicator() : Text("🚀 Enhance")),
              ),
            ],
          ),

          if (psnr!=null)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text("PSNR ${psnr!.toStringAsFixed(2)} | SSIM ${ssim!.toStringAsFixed(3)} | MSE ${mse!.toStringAsFixed(1)}"),
            ),
        ],
      ),
    );
  }

  Widget _panel(String label, Uint8List? img) {
    return Expanded(
      child: Container(
        decoration: BoxDecoration(border: Border.all(color: Colors.blueAccent, width: 2)),
        child: img == null
            ? Center(child: Text(label))
            : Image.memory(img, fit: BoxFit.contain),
      ),
    );
  }
}
