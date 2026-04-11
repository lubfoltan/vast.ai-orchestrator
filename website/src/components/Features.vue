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
  { icon: '🖥️', title: 'Desktop GUI', desc: 'Dark-mode CustomTkinter interface with scrollable config, real-time logs, and SSH console.' },
  { icon: '⚡', title: 'One-Click Pipeline', desc: 'Rent GPU → upload data → train → download results. Fully automated 5-step pipeline.' },
  { icon: '🔀', title: 'Classification & Regression', desc: 'Switch between image classification (CNN) and tabular regression (MLP) with a single toggle.' },
  { icon: '📊', title: '17 Metrics', desc: '10 classification metrics (AUC, F1, Cohen\'s Kappa…) and 7 regression metrics (RMSE, R², MAPE…).' },
  { icon: '📜', title: 'Custom Scripts', desc: 'Upload and run your own training script instead of the built-in one — full flexibility.' },
  { icon: '🔑', title: 'SSH Key Auth', desc: 'Authenticate with SSH key or password. Interactive SSH console available in the GUI.' },
  { icon: '📁', title: 'Auto-Organize', desc: 'Flat folder of images? Regex-based auto-organize sorts them into class sub-folders.' },
  { icon: '📈', title: 'Excel Reports', desc: 'Results exported to results.xlsx with epoch history, final metrics, and prediction sheets.' },
  { icon: '🎛️', title: 'Training Flags', desc: 'Early stopping, LR scheduler, Mixup augmentation, label smoothing, Grad-CAM — all toggleable.' },
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
