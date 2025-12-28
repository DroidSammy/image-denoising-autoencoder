import 'package:flutter/material.dart';
import 'screens/home_page.dart';

void main() {
  runApp(const DenoiseApp());
}

class DenoiseApp extends StatelessWidget {
  const DenoiseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Image Denoising",
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.blue,
      ),
      home: const HomePage(),
    );
  }
}
