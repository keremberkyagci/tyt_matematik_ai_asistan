/// Sohbet ekranındaki her bir mesajı temsil eden veri modelidir.
class Message {
  /// Mesajın metin içeriği. 
  /// Akışlı (streaming) güncelleme yapılabilmesi için 'final' değil.
  String text;
  
  /// Mesajın kullanıcıya mı (true) yoksa yapay zekaya mı (false) ait olduğu bilgisi.
  final bool isUser;
  
  /// Mesajın gönderilme veya alınma zamanı.
  final DateTime timestamp;

  Message({
    required this.text,
    required this.isUser,
    required this.timestamp,
  });

  /// Mesajı JSON formatına dönüştürür (Eğer sunucuya mesaj göndermek gerekirse).
  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'is_user': isUser,
    };
  }

  /// Sunucudan gelen JSON verisini Message nesnesine dönüştürür.
  factory Message.fromJson(Map<String, dynamic> json) {
    return Message(
      text: json['text'] ?? '',
      isUser: json['is_user'] ?? false,
      timestamp: DateTime.now(),
    );
  }
}
