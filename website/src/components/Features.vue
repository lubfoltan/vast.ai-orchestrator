<template>
  <section id="features" class="section">
    <h2 class="section-title">Features</h2>
    <p class="section-sub">Everything you need, nothing you don't.</p>
    <div class="grid">
      <div class="card" v-for="f in features" :key="f.title">
        <div class="card-icon">{{ f.icon }}</div>
        <h3>{{ f.title }}</h3>
        <p>{{ f.desc }}</p>
      </div>
    </div>
  </section>
</template>

<script setup>
const features = [
  { icon: '🧠', title: 'Smart Estimator', desc: 'Pre-flight analysis of your dataset — estimates VRAM needs, training time, and detects class imbalance before you spend a penny.' },
  { icon: '⚡', title: 'Dynamic Provisioning', desc: 'Scores GPUs by value — factors in reliability, network speed, and DL performance, not just price.' },
  { icon: '📈', title: 'Live Telemetry', desc: 'Real-time Loss/Accuracy charts in the GUI via periodic CSV polling. No W&B dependency needed.' },
  { icon: '🔄', title: 'Rsync Pipeline', desc: 'Resume-capable, checksum-verified file transfers. Falls back to tar+gzip → SFTP automatically.' },
  { icon: '🔬', title: 'Automated Grad-CAM', desc: 'Generates 5 Class Activation Map heatmaps after training — shows what the model sees, not just numbers.' },
  { icon: '🔀', title: 'Classification & Regression', desc: 'Image classification with pretrained CNNs or tabular regression with MLP — switch with a toggle.' },
  { icon: '📊', title: '17 Metrics + Reports', desc: '10 classification + 7 regression metrics. Auto-generated REPORT.md and Excel with history, metrics, and predictions.' },
  { icon: '📜', title: 'Custom Scripts', desc: 'Upload and run your own training script instead of the built-in one — full flexibility with guided documentation.' },
  { icon: '🎛️', title: 'Training Flags', desc: 'Early stopping, LR scheduler, Mixup augmentation, label smoothing, Grad-CAM — all toggleable from the GUI.' },
]
</script>

<style scoped>
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}
.card{background:var(--bg2);border:1px solid var(--bg3);border-radius:14px;padding:28px;transition:border-color .2s}
.card:hover{border-color:var(--accent)}
.card-icon{font-size:1.6rem;margin-bottom:12px}
.card h3{font-size:1.05rem;font-weight:600;margin-bottom:8px}
.card p{color:var(--muted);font-size:.9rem;line-height:1.55}
</style>
