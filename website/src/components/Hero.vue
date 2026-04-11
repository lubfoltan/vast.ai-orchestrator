<template>
  <section class="hero">
    <div class="hero-inner">
      <span class="badge badge-blue">Open Source</span>
      <h1>SCOUT<br><span class="gradient">Training Orchestrator</span></h1>
      <p class="tagline">Configure. Deploy. Harvest.</p>
      <p class="hero-sub">
        One-click ML training on cloud GPUs. Desktop app that rents a
        <a href="https://vast.ai" target="_blank">Vast.ai</a> instance, uploads
        your data, trains, and downloads results&nbsp;&mdash; fully automated.
      </p>
      <div class="hero-buttons">
        <a href="https://github.com/your-username/vast-ai-orchestrator" target="_blank" class="btn btn-primary">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .3a12 12 0 00-3.8 23.38c.6.11.82-.26.82-.58v-2.16c-3.34.73-4.04-1.61-4.04-1.61a3.18 3.18 0 00-1.33-1.76c-1.09-.74.08-.73.08-.73a2.52 2.52 0 011.84 1.24 2.56 2.56 0 003.5 1 2.56 2.56 0 01.76-1.61c-2.67-.3-5.47-1.33-5.47-5.93a4.64 4.64 0 011.24-3.22 4.3 4.3 0 01.12-3.18s1-.32 3.3 1.23a11.38 11.38 0 016 0c2.28-1.55 3.28-1.23 3.28-1.23a4.3 4.3 0 01.12 3.18 4.64 4.64 0 011.24 3.22c0 4.61-2.81 5.63-5.48 5.92a2.87 2.87 0 01.82 2.23v3.29c0 .32.21.7.82.58A12 12 0 0012 .3"/></svg>
          GitHub
        </a>
        <a href="#features" class="btn btn-outline">See Features</a>
      </div>
      <div class="hero-tech">
        <span>Python</span><span>CustomTkinter</span><span>PyTorch</span><span>Paramiko</span><span>Vast.ai SDK</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.hero{
  min-height:90vh;display:flex;align-items:center;justify-content:center;
  text-align:center;padding:60px 24px 80px;
  background:radial-gradient(ellipse at 50% 0%,rgba(59,130,246,.12) 0%,transparent 60%);
}
.hero-inner{max-width:720px}
h1{font-size:clamp(2.4rem,5vw,3.8rem);font-weight:800;line-height:1.15;margin:16px 0 20px}
.gradient{background:linear-gradient(135deg,var(--accent),var(--accent3));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-sub{font-size:1.15rem;color:var(--muted);max-width:560px;margin:0 auto 32px}
.tagline{font-size:1.3rem;font-weight:600;color:var(--accent3);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.hero-buttons{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 24px;border-radius:10px;font-weight:600;font-size:.95rem;transition:all .2s}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover{background:#2563eb;text-decoration:none}
.btn-outline{border:1px solid var(--muted);color:var(--text)}
.btn-outline:hover{border-color:var(--accent);color:var(--accent);text-decoration:none}
.hero-tech{margin-top:48px;display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
.hero-tech span{padding:6px 14px;border-radius:8px;background:var(--bg3);font-size:.82rem;color:var(--muted);font-weight:500}
</style>
