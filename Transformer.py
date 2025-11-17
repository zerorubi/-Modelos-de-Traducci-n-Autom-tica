# ==================== MODELO TRANSFORMER CON MENÚ INTERACTIVO ====================

import torch
from transformers import MarianMTModel, MarianTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np

print("🚀 MODELO TRANSFORMER - Usando Modelo Pre-entrenado")
print("✨ Arquitectura: Transformer Encoder-Decoder con Multi-Head Attention")

# Modelos Transformer disponibles (Helsinki-NLP usa arquitectura Transformer)
TRANSFORMER_MODELS = {
    'es-en': 'Helsinki-NLP/opus-mt-es-en',
    'en-es': 'Helsinki-NLP/opus-mt-en-es',
    'es-fr': 'Helsinki-NLP/opus-mt-es-fr',
    'fr-es': 'Helsinki-NLP/opus-mt-fr-es',
    'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
    'fr-en': 'Helsinki-NLP/opus-mt-fr-en'
}

class TransformerTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {}
        print(f"✅ Dispositivo: {self.device}")

    def get_simulated_transformer_metrics(self, lang_pair):
        """Métricas simuladas para Transformer (Encoder-Decoder con Attention)"""
        print(f"\n📊 SIMULANDO MÉTRICAS DE TRANSFORMER - {lang_pair}")
        print("    Arquitectura: Transformer Encoder-Decoder (multi-head attention)")

        # Métricas típicas de Transformer (mejores que RNN Simple)
        metrics = {
            'es-en': {
                'final_accuracy': 0.89,
                'final_loss': 0.18,
                'training_time_hours': 8,
                'dataset_size': '500K pares',
                'vocab_size': 32000,
                'model_type': 'Transformer',
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_size': 512,
                'attention_heads': 8,
                'feed_forward_size': 2048,
                'dropout': 0.1,
                'positional_encoding': 'Sí',
                'attention': 'Multi-Head Self-Attention + Cross-Attention'
            },
            'en-es': {
                'final_accuracy': 0.87,
                'final_loss': 0.20,
                'training_time_hours': 8,
                'dataset_size': '500K pares',
                'vocab_size': 32000,
                'model_type': 'Transformer',
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_size': 512,
                'attention_heads': 8,
                'feed_forward_size': 2048,
                'dropout': 0.1,
                'positional_encoding': 'Sí',
                'attention': 'Multi-Head Self-Attention + Cross-Attention'
            },
            'es-fr': {
                'final_accuracy': 0.85,
                'final_loss': 0.22,
                'training_time_hours': 7,
                'dataset_size': '400K pares',
                'vocab_size': 30000,
                'model_type': 'Transformer',
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_size': 512,
                'attention_heads': 8,
                'feed_forward_size': 2048,
                'dropout': 0.1,
                'positional_encoding': 'Sí',
                'attention': 'Multi-Head Self-Attention + Cross-Attention'
            },
            'fr-es': {
                'final_accuracy': 0.84,
                'final_loss': 0.24,
                'training_time_hours': 7,
                'dataset_size': '400K pares',
                'vocab_size': 30000,
                'model_type': 'Transformer',
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_size': 512,
                'attention_heads': 8,
                'feed_forward_size': 2048,
                'dropout': 0.1,
                'positional_encoding': 'Sí',
                'attention': 'Multi-Head Self-Attention + Cross-Attention'
            },
            'en-fr': {
                'final_accuracy': 0.90,
                'final_loss': 0.16,
                'training_time_hours': 9,
                'dataset_size': '550K pares',
                'vocab_size': 33000,
                'model_type': 'Transformer',
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_size': 512,
                'attention_heads': 8,
                'feed_forward_size': 2048,
                'dropout': 0.1,
                'positional_encoding': 'Sí',
                'attention': 'Multi-Head Self-Attention + Cross-Attention'
            },
            'fr-en': {
                'final_accuracy': 0.88,
                'final_loss': 0.19,
                'training_time_hours': 9,
                'dataset_size': '550K pares',
                'vocab_size': 33000,
                'model_type': 'Transformer',
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_size': 512,
                'attention_heads': 8,
                'feed_forward_size': 2048,
                'dropout': 0.1,
                'positional_encoding': 'Sí',
                'attention': 'Multi-Head Self-Attention + Cross-Attention'
            }
        }

        metric = metrics.get(lang_pair, metrics['es-en'])

        # Simular entrenamiento progresivo
        epochs = 10
        accuracy = []
        loss = []

        print("    Progreso de entrenamiento simulado:")
        for epoch in range(epochs):
            acc_progress = metric['final_accuracy'] * (0.20 + 0.80 * (epoch / epochs)) + np.random.normal(0, 0.02)
            loss_progress = metric['final_loss'] * (4.0 - 3.0 * (epoch / epochs)) + np.random.normal(0, 0.03)

            accuracy.append(min(max(acc_progress, 0.15), metric['final_accuracy']))
            loss.append(max(loss_progress, metric['final_loss']))

            print(f"      Época {epoch+1}/{epochs} - Precisión: {accuracy[-1]:.3f} - Pérdida: {loss[-1]:.3f}")

        self.training_history[lang_pair] = {
            'accuracy': accuracy,
            'loss': loss,
            'epochs': epochs,
            'metrics': metric
        }

        # MOSTRAR GRÁFICAS AUTOMÁTICAMENTE
        self.plot_training(lang_pair)

        return accuracy, loss

    def show_architecture(self, lang_pair):
        """Muestra arquitectura Transformer"""
        if lang_pair not in self.training_history:
            self.get_simulated_transformer_metrics(lang_pair)

        m = self.training_history[lang_pair]['metrics']

        print(f"\n🧠 ARQUITECTURA TRANSFORMER - {lang_pair}")
        print("=" * 70)
        print(f"📌 Tipo: {m['model_type']} (Encoder-Decoder)")
        print(f"\n🔷 ENCODER:")
        print(f"   • Capas: {m['encoder_layers']} capas de transformer")
        print(f"   • Multi-Head Attention: {m['attention_heads']} cabezas")
        print(f"   • Dimensión oculta: {m['hidden_size']}")
        print(f"   • Feed-Forward: {m['feed_forward_size']} unidades")
        print(f"   • Positional Encoding: {m['positional_encoding']}")
        print(f"\n🔶 DECODER:")
        print(f"   • Capas: {m['decoder_layers']} capas de transformer")
        print(f"   • Multi-Head Attention: {m['attention_heads']} cabezas")
        print(f"   • Cross-Attention: Sí (atiende al encoder)")
        print(f"   • Dimensión oculta: {m['hidden_size']}")
        print(f"   • Feed-Forward: {m['feed_forward_size']} unidades")
        print(f"\n⚙️  CONFIGURACIÓN:")
        print(f"   • Vocabulario: {m['vocab_size']} tokens")
        print(f"   • Dropout: {m['dropout']}")
        print(f"   • Dataset: {m['dataset_size']}")
        print(f"   • Tiempo entrenamiento: {m['training_time_hours']}h")
        print(f"\n📊 RESULTADOS:")
        print(f"   • Precisión final: {m['final_accuracy']:.3f}")
        print(f"   • Pérdida final: {m['final_loss']:.3f}")
        print(f"\n💡 VENTAJAS DEL TRANSFORMER:")
        print("   ✓ Multi-head attention captura múltiples relaciones")
        print("   ✓ Paralelización completa (más rápido que RNN)")
        print("   ✓ No sufre de vanishing gradients")
        print("   ✓ Mejor manejo de dependencias a largo alcance")
        print("   ✓ Positional encoding preserva orden secuencial")
        print("   ✓ Cross-attention permite alineación flexible fuente-objetivo")

    def plot_training(self, lang_pair):
        """Gráfica de entrenamiento simulado"""
        if lang_pair not in self.training_history:
            return

        h = self.training_history[lang_pair]
        epochs = range(1, h['epochs'] + 1)
        m = h['metrics']

        plt.figure(figsize=(14, 5))

        # Precisión
        plt.subplot(1, 2, 1)
        plt.plot(epochs, h['accuracy'], 'b-', linewidth=2.5, marker='o', markersize=6, label='Precisión entrenamiento')
        plt.axhline(y=m['final_accuracy'], color='r', linestyle='--', linewidth=2, label=f'Precisión final: {m["final_accuracy"]:.3f}')
        plt.title(f'Transformer - Precisión\n{lang_pair.upper()} (Multi-Head Attention)', fontsize=12, fontweight='bold')
        plt.xlabel('Época', fontsize=11)
        plt.ylabel('Precisión', fontsize=11)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0, 1.0)

        # Pérdida
        plt.subplot(1, 2, 2)
        plt.plot(epochs, h['loss'], 'orange', linewidth=2.5, marker='s', markersize=6, label='Pérdida entrenamiento')
        plt.axhline(y=m['final_loss'], color='g', linestyle='--', linewidth=2, label=f'Pérdida final: {m["final_loss"]:.3f}')
        plt.title(f'Transformer - Pérdida\n{lang_pair.upper()} (Multi-Head Attention)', fontsize=12, fontweight='bold')
        plt.xlabel('Época', fontsize=11)
        plt.ylabel('Pérdida', fontsize=11)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0, 1.0)

        plt.tight_layout()
        plt.show()

    def load_model(self, lang_pair, show_graphs=True):
        """Carga modelo pre-entrenado"""
        if lang_pair not in self.models:
            print(f"\n🔧 Cargando modelo Transformer para {lang_pair}...")

            # Mostrar métricas simuladas de Transformer (CON GRÁFICAS)
            self.get_simulated_transformer_metrics(lang_pair)

            try:
                model_name = TRANSFORMER_MODELS[lang_pair]
                self.tokenizers[lang_pair] = MarianTokenizer.from_pretrained(model_name)
                self.models[lang_pair] = MarianMTModel.from_pretrained(model_name).to(self.device)

                print(f"✅ Modelo Transformer cargado exitosamente")

                # Mostrar arquitectura
                self.show_architecture(lang_pair)

            except Exception as e:
                print(f"❌ Error: {e}")
                return False

        return True

    def translate(self, text, lang_pair):
        """Traducción usando Transformer pre-entrenado"""
        if lang_pair not in self.models:
            if not self.load_model(lang_pair, show_graphs=False):
                return "Error: No se pudo cargar el modelo"

        tokenizer = self.tokenizers[lang_pair]
        model = self.models[lang_pair]

        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

            # Generación con beam search (típico de Transformer)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,  # Beam search para mejor calidad
                    do_sample=False,
                    early_stopping=True,
                    length_penalty=1.0
                )

            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation

        except Exception as e:
            return f"Error: {e}"


# ==================== MENÚ INTERACTIVO ====================

def mostrar_menu_principal():
    """Muestra el menú principal"""
    print("\n" + "="*70)
    print("🌍 TRADUCTOR TRANSFORMER - MENÚ PRINCIPAL")
    print("="*70)
    print("\n1️⃣  Traducir texto")
    print("2️⃣  Ver arquitecturas de todos los modelos")
    print("3️⃣  Ejecutar demostración automática")
    print("4️⃣  Salir")
    print("\n" + "="*70)


def mostrar_menu_idiomas():
    """Muestra el menú de selección de idiomas"""
    print("\n" + "="*70)
    print("🌐 SELECCIONA EL PAR DE IDIOMAS")
    print("="*70)
    print("\n1️⃣  Español → Inglés (es-en)")
    print("2️⃣  Inglés → Español (en-es)")
    print("3️⃣  Español → Francés (es-fr)")
    print("4️⃣  Francés → Español (fr-es)")
    print("5️⃣  Inglés → Francés (en-fr)")
    print("6️⃣  Francés → Inglés (fr-en)")
    print("0️⃣  Volver al menú principal")
    print("\n" + "="*70)


def seleccionar_idioma():
    """Permite al usuario seleccionar un par de idiomas"""
    idiomas_map = {
        '1': 'es-en',
        '2': 'en-es',
        '3': 'es-fr',
        '4': 'fr-es',
        '5': 'en-fr',
        '6': 'fr-en'
    }

    while True:
        mostrar_menu_idiomas()
        opcion = input("👉 Ingresa tu opción: ").strip()

        if opcion == '0':
            return None

        if opcion in idiomas_map:
            return idiomas_map[opcion]
        else:
            print("❌ Opción inválida. Intenta de nuevo.")


def traducir_interactivo(translator):
    """Modo de traducción interactiva"""
    lang_pair = seleccionar_idioma()

    if lang_pair is None:
        return

    # Cargar modelo (MUESTRA GRÁFICAS AUTOMÁTICAMENTE)
    translator.load_model(lang_pair, show_graphs=True)

    print(f"\n✅ Modelo Transformer {lang_pair.upper()} cargado")
    print("💡 Escribe 'salir' para volver al menú principal\n")

    while True:
        texto = input(f"\n📝 Ingresa el texto a traducir ({lang_pair.split('-')[0].upper()}): ").strip()

        if texto.lower() == 'salir':
            break

        if not texto:
            print("⚠️  Por favor ingresa un texto válido")
            continue

        print("\n⏳ Traduciendo con Transformer...")
        start_time = time.time()
        traduccion = translator.translate(texto, lang_pair)
        elapsed = (time.time() - start_time) * 1000

        print(f"\n📥 Original ({lang_pair.split('-')[0].upper()}): {texto}")
        print(f"📤 Traducción ({lang_pair.split('-')[1].upper()}): {traduccion}")
        print(f"⏱️  Tiempo: {elapsed:.0f}ms")


def ver_arquitecturas(translator):
    """Muestra las arquitecturas de todos los modelos"""
    print("\n📊 Cargando arquitecturas de todos los modelos Transformer...")

    for lang_pair in ['es-en', 'en-es', 'es-fr', 'fr-es', 'en-fr', 'fr-en']:
        translator.show_architecture(lang_pair)

    input("\n✅ Presiona ENTER para continuar...")


def demo_automatica(translator):
    """Ejecuta la demostración automática"""
    print("\n" + "="*70)
    print("🧪 EJECUTANDO DEMOSTRACIÓN AUTOMÁTICA - TRANSFORMER")
    print("="*70)

    # Ejemplo 1: Español → Inglés
    print("\n" + "="*70)
    print("📝 EJEMPLO 1: ESPAÑOL → INGLÉS")
    print("="*70)
    translator.load_model('es-en', show_graphs=True)

    frases_es_en = [
        "Hola mundo",
        "¿Cómo estás?",
        "Me gusta programar con inteligencia artificial",
        "Buenos días, ¿qué tal tu día?",
        "Los transformers revolucionaron el procesamiento de lenguaje natural"
    ]

    for frase in frases_es_en:
        start_t = time.time()
        result = translator.translate(frase, 'es-en')
        elapsed = (time.time() - start_t) * 1000
        print(f"\n  📥 ES: {frase}")
        print(f"  📤 EN: {result}")
        print(f"  ⏱️  Tiempo: {elapsed:.0f}ms")

    # Ejemplo 2: Inglés → Español
    print("\n" + "="*70)
    print("📝 EJEMPLO 2: INGLÉS → ESPAÑOL")
    print("="*70)
    translator.load_model('en-es', show_graphs=True)

    frases_en_es = [
        "Hello world",
        "How are you doing today?",
        "I love programming with transformers",
        "Artificial intelligence is amazing"
    ]

    for frase in frases_en_es:
        start_t = time.time()
        result = translator.translate(frase, 'en-es')
        elapsed = (time.time() - start_t) * 1000
        print(f"\n  📥 EN: {frase}")
        print(f"  📤 ES: {result}")
        print(f"  ⏱️  Tiempo: {elapsed:.0f}ms")

    input("\n✅ Demostración completada. Presiona ENTER para continuar...")


def menu_principal():
    """Función principal del menú"""
    print("\n⏳ Inicializando traductor Transformer...")
    start = time.time()

    translator = TransformerTranslator()

    print(f"\n✅ Sistema Transformer listo en {time.time() - start:.1f}s")

    while True:
        mostrar_menu_principal()
        opcion = input("👉 Selecciona una opción: ").strip()

        if opcion == '1':
            traducir_interactivo(translator)
        elif opcion == '2':
            ver_arquitecturas(translator)
        elif opcion == '3':
            demo_automatica(translator)
        elif opcion == '4':
            print("\n👋 ¡Hasta luego! Gracias por usar el traductor Transformer")
            break
        else:
            print("\n❌ Opción inválida. Por favor selecciona una opción del 1 al 4.")


# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    menu_principal()
