import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/chat_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    const navyBlue = Color(0xFF1A237E);

    return MaterialApp(
      title: 'TYY MATEMATİK ASİSTANI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: navyBlue,
          primary: navyBlue,
          onPrimary: Colors.white,
          surface: navyBlue,
          onSurface: Colors.white,
        ),
        scaffoldBackgroundColor: navyBlue,
        fontFamily: GoogleFonts.oswald().fontFamily,
        textTheme: GoogleFonts.oswaldTextTheme(
          ThemeData.light().textTheme.copyWith(
            bodyLarge: const TextStyle(color: Colors.white),
            bodyMedium: const TextStyle(color: Colors.white),
          ),
        ),
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}
