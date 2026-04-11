<template>
  <section class="section">
    <h2 class="section-title">How It Works</h2>
    <p class="section-sub">Six steps from config to trained model.</p>
    <div class="steps">
      <div class="step" v-for="(s, i) in steps" :key="i">
        <div class="step-num">{{ i + 1 }}</div>
        <div>
          <h3>{{ s.title }}</h3>
          <p>{{ s.desc }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const steps = [
  { title: 'Estimate', desc: 'Smart Estimator scans your dataset — file count, size, class balance — and estimates VRAM needs and training time.' },
  { title: 'Provision', desc: 'Dynamic scoring finds the best-value GPU — filters by reliability, network speed, and compute power, not just price.' },
  { title: 'Upload', desc: 'Dataset transferred via rsync (resume + checksum). Falls back to tar+gzip archive for maximum compatibility.' },
  { title: 'Train', desc: 'Training runs remotely with live log streaming and a real-time Loss/Accuracy chart in the GUI.' },
  { title: 'Interpret', desc: 'Grad-CAM heatmaps generated automatically. REPORT.md and results.xlsx created on the server.' },
  { title: 'Harvest', desc: 'Model weights, plots, metrics, Excel report, and Markdown report downloaded. Instance destroyed to stop billing.' },
]
</script>

<style scoped>
.steps{max-width:640px;margin:0 auto;display:flex;flex-direction:column;gap:28px}
.step{display:flex;gap:20px;align-items:flex-start}
.step-num{
  flex-shrink:0;width:40px;height:40px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:1rem;color:#fff;
}
.step h3{font-size:1rem;font-weight:600;margin-bottom:4px}
.step p{color:var(--muted);font-size:.9rem}
</style>
