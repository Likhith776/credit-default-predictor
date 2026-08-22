/* Credit Default Predictor — frontend interactions
 * GSAP + ScrollTrigger for scroll-driven motion, Lenis for inertial scroll.
 * All motion uses transform/opacity only. prefers-reduced-motion disables
 * Lenis, pinning, and scrubbed timelines in favor of simple fades.
 */
(() => {
  "use strict";

  gsap.registerPlugin(ScrollTrigger);

  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => Array.from(c.querySelectorAll(s));

  /* ------------------------------------------------ Lenis smooth scroll */
  let lenis = null;
  if (!REDUCED && typeof Lenis !== "undefined") {
    lenis = new Lenis({ duration: 1.1, smoothWheel: true });
    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((t) => lenis.raf(t * 1000));
    gsap.ticker.lagSmoothing(0);
  }

  const scrollTo = (target) => {
    const el = typeof target === "string" ? $(target) : target;
    if (!el) return;
    if (lenis) lenis.scrollTo(el, { offset: -70 });
    else el.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" });
  };

  $$("[data-scroll-to]").forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      scrollTo(a.dataset.scrollTo);
      $("#navLinks").classList.remove("is-open");
      $("#navToggle").classList.remove("is-open");
    })
  );

  /* ------------------------------------------------ nav state */
  const nav = $("#nav");
  const onScrollNav = () => nav.classList.toggle("is-scrolled", window.scrollY > 40);
  window.addEventListener("scroll", onScrollNav, { passive: true });
  onScrollNav();

  const navToggle = $("#navToggle");
  navToggle.addEventListener("click", () => {
    const open = $("#navLinks").classList.toggle("is-open");
    navToggle.classList.toggle("is-open", open);
    navToggle.setAttribute("aria-expanded", String(open));
  });

  // active-section highlighting (throttled via ScrollTrigger, not scroll events)
  $$('main section[id]').forEach((sec) => {
    ScrollTrigger.create({
      trigger: sec,
      start: "top 55%",
      end: "bottom 55%",
      onToggle: (self) => {
        if (!self.isActive) return;
        $$(".nav-links a[data-scroll-to]").forEach((a) =>
          a.classList.toggle("is-active", a.dataset.scrollTo === `#${sec.id}`)
        );
      },
    });
  });

  /* ------------------------------------------------ hero ticker */
  const TICKS = [
    "EXT_SOURCE_MEAN", "CREDIT_TERM", "RANK_CREDIT_INCOME_RATIO_BY_INCOME_TYPE",
    "CC_UTILIZATION_TREND", "BUREAU_DPD_DIFF_MEAN", "INSTAL_LATE_RATIO",
    "PREV_APPROVAL_RATE", "DAYS_EMPLOYED_RATIO", "POS_SK_DPD_TREND",
  ];
  const track = $("#tickerTrack");
  if (track) {
    track.innerHTML = [...TICKS, ...TICKS].map((t) => `<span>${t}</span>`).join("");
    if (!REDUCED) {
      gsap.to(track, { xPercent: -50, duration: 30, ease: "none", repeat: -1 });
    }
  }

  /* ------------------------------------------------ generic reveals */
  const revealTargets = $$("[data-reveal]");
  if (REDUCED) {
    revealTargets.forEach((el) => (el.style.opacity = 1));
  } else {
    revealTargets.forEach((el) => {
      gsap.fromTo(el,
        { y: 34, opacity: 0 },
        {
          y: 0, opacity: 1, duration: 0.9, ease: "power3.out",
          scrollTrigger: { trigger: el, start: "top 88%", once: true },
        }
      );
    });
  }

  /* ------------------------------------------------ count-ups */
  $$("[data-count]").forEach((el) => {
    const end = parseFloat(el.dataset.count);
    const fmt = (v) => Math.round(v).toLocaleString("en-US");
    if (REDUCED) { el.textContent = fmt(end); return; }
    const obj = { v: 0 };
    gsap.to(obj, {
      v: end, duration: 1.8, ease: "power2.out",
      scrollTrigger: { trigger: el, start: "top 90%", once: true },
      onUpdate: () => (el.textContent = fmt(obj.v)),
    });
  });
  $$("[data-count-dec]").forEach((el) => {
    const end = parseFloat(el.dataset.countDec);
    if (REDUCED) { el.textContent = end.toFixed(3); return; }
    const obj = { v: 0 };
    gsap.to(obj, {
      v: end, duration: 1.8, ease: "power2.out",
      scrollTrigger: { trigger: el, start: "top 90%", once: true },
      onUpdate: () => (el.textContent = obj.v.toFixed(3)),
    });
  });

  /* ------------------------------------------------ HOW IT WORKS pinned scrub */
  const STAGES = ["INTAKE", "FEATURES", "SCORING", "EXPLAIN", "VERDICT"];
  const MOCK_FACTORS = [
    { name: "EXT_SOURCE_MEAN", impact: -0.62 },
    { name: "CREDIT_TERM", impact: 0.41 },
    { name: "DAYS_BIRTH", impact: 0.28 },
    { name: "DAYS_EMPLOYED_RATIO", impact: -0.22 },
    { name: "ANNUITY_INCOME_RATIO", impact: 0.18 },
    { name: "CC_UTILIZATION_TREND", impact: 0.12 },
  ];

  // build mock waterfall bars
  const wfBars = $("#wfBars");
  const maxAbs = Math.max(...MOCK_FACTORS.map((f) => Math.abs(f.impact)));
  MOCK_FACTORS.forEach((f) => {
    const row = document.createElement("div");
    row.className = "wf-row";
    const w = (Math.abs(f.impact) / maxAbs) * 48; // percent of half-track
    row.innerHTML = `
      <div class="wf-name">${f.name}</div>
      <div class="wf-track">
        <div class="wf-bar ${f.impact > 0 ? "pos" : "neg"}" data-w="${w}"></div>
      </div>`;
    wfBars.appendChild(row);
  });

  // build fake trees
  const lbTrees = $("#lbTrees");
  const treeHeights = Array.from({ length: 28 }, (_, i) => 22 + 66 * Math.abs(Math.sin(i * 1.7)));
  lbTrees.innerHTML = treeHeights.map((h) => `<i style="height:${h}%"></i>`).join("");

  const stageEls = {
    intake: $("#intakeGrid"), flow: $("#flowLines"), features: $("#featureBox"),
    lgbm: $("#lgbmBox"), waterfall: $("#waterfallMock"), verdict: $("#verdictBox"),
  };

  function setStage(i) {
    const caption = $("#stageCaption");
    caption.textContent = `STAGE 0${i + 1} — ${STAGES[i]}`;
    $$(".step").forEach((s) => s.classList.toggle("is-active", +s.dataset.step === i));

    const show = (el, on) =>
      el && gsap.to(el, {
        autoAlpha: on ? 1 : 0, duration: REDUCED ? 0 : 0.45, ease: "power2.out", overwrite: true,
      });
    show(stageEls.intake, i === 0);
    show(stageEls.flow, i === 0);
    show(stageEls.features, i === 1);
    show(stageEls.lgbm, i === 2);
    show(stageEls.waterfall, i === 3);
    show(stageEls.verdict, i === 4);

    if (i === 2 && !REDUCED) {
      gsap.fromTo("#lbTrees i", { scaleY: 0 }, { scaleY: 1, duration: 0.5, stagger: 0.02, ease: "power2.out", overwrite: true });
    }
    if (i === 3 && !REDUCED) {
      gsap.to("#wfBars .wf-bar", {
        scaleX: (idx, el) => parseFloat(el.dataset.w) / 48,
        duration: 0.55, stagger: 0.06, ease: "power2.out", overwrite: true,
      });
    } else {
      gsap.set("#wfBars .wf-bar", { scaleX: (idx, el) => (i > 3 ? parseFloat(el.dataset.w) / 48 : 0) });
    }
    if (i === 4) {
      const target = { v: 0 };
      const vb = { val: $("#vbValue"), tier: $("#vbTier") };
      if (REDUCED) { vb.val.textContent = "0.421"; vb.tier.textContent = "MODERATE RISK"; return; }
      gsap.to(target, {
        v: 0.421, duration: 0.8, ease: "power2.out", overwrite: true,
        onUpdate: () => (vb.val.textContent = target.v.toFixed(3)),
        onComplete: () => (vb.tier.textContent = "MODERATE RISK"),
      });
    }
  }

  const mm = gsap.matchMedia();

  // Desktop / tablet: pin the visual, scrub steps.
  mm.add("(min-width: 900px) and (prefers-reduced-motion: no-preference)", () => {
    const visual = $("#howVisual");
    ScrollTrigger.create({
      trigger: "#howPin",
      start: "top top+=120",
      end: "bottom bottom-=140",
      pin: visual,
      pinSpacing: true,
      anticipatePin: 1,
    });

    $$(".step").forEach((stepEl, i) => {
      ScrollTrigger.create({
        trigger: stepEl,
        start: "top 55%",
        end: "bottom 55%",
        onEnter: () => setStage(i),
        onEnterBack: () => setStage(i),
      });
    });
    setStage(0);
  });

  // Mobile / reduced motion: no pinning — sequential reveals, tap-free auto stages.
  mm.add("(max-width: 899px), (prefers-reduced-motion: reduce)", () => {
    $$(".step").forEach((stepEl, i) => {
      ScrollTrigger.create({
        trigger: stepEl,
        start: "top 75%",
        onEnter: () => setStage(i),
        onEnterBack: () => setStage(i),
      });
    });
    if (REDUCED) {
      // everything visible, no animation
      gsap.set(".stage-body > div", { autoAlpha: 0 });
      setStage(0);
    }
  });

  /* ------------------------------------------------ predict form */
  const form = $("#predictForm");
  const status = $("#formStatus");
  const gaugeValue = $("#gaugeValue");
  const gaugeTier = $("#gaugeTier");
  const gaugeNeedle = $("#gaugeNeedle");
  const wfReal = $("#wfRealBars");

  // live slider value labels
  $("#f-employed").addEventListener("input", (e) =>
    ($("#v-employed").textContent = `${parseFloat(e.target.value).toFixed(1)} yrs`));
  $("#f-age").addEventListener("input", (e) =>
    ($("#v-age").textContent = e.target.value));

  function fmtVal(v) {
    if (Math.abs(v) >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
    if (Math.abs(v) >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }

  function animateGauge(prob) {
    // needle sweeps -90deg (0%) to +90deg (100%), eased — no snap
    const deg = -90 + prob * 180;
    if (REDUCED) {
      gaugeNeedle.style.transform = `rotate(${deg}deg)`;
    } else {
      gsap.to(gaugeNeedle, { rotation: deg, duration: 1.1, ease: "power3.out", overwrite: true });
    }
    const tier = prob < 0.2 ? "LOW RISK" : prob < 0.5 ? "MODERATE RISK" : "HIGH RISK";
    gaugeTier.textContent = tier;
    gaugeTier.className = `gauge-tier mono t-${tier.split(" ")[0].toLowerCase()}`;
    if (REDUCED) {
      gaugeValue.textContent = `${(prob * 100).toFixed(1)}%`;
      return;
    }
    const counter = { v: 0 };
    gsap.to(counter, {
      v: prob * 100, duration: 1.1, ease: "power3.out", overwrite: true,
      onUpdate: () => (gaugeValue.textContent = `${counter.v.toFixed(1)}%`),
    });
  }

  function renderWaterfall(factors) {
    const max = Math.max(...factors.map((f) => Math.abs(f.impact))) || 1;
    wfReal.innerHTML = factors
      .map((f, i) => {
        const w = (Math.abs(f.impact) / max) * 46;
        const cls = f.impact > 0 ? "pos" : "neg";
        const arrow = f.impact > 0 ? "▲" : "▼";
        return `
        <div class="wfr-row">
          <div class="wfr-name">${f.feature}<small>${arrow} ${fmtVal(f.value)}</small></div>
          <div class="wfr-track"><div class="wfr-bar ${cls}" data-w="${w}"></div></div>
        </div>`;
      })
      .join("");
    if (REDUCED) {
      $$(".wfr-bar", wfReal).forEach((b) => (b.style.transform = `scaleX(${b.dataset.w / 46})`));
      return;
    }
    gsap.fromTo($$(".wfr-row", wfReal),
      { opacity: 0, x: 18 },
      { opacity: 1, x: 0, duration: 0.45, stagger: 0.06, ease: "power2.out" }
    );
    gsap.fromTo($$(".wfr-bar", wfReal),
      { scaleX: 0 },
      { scaleX: (i, el) => parseFloat(el.dataset.w) / 46, duration: 0.6, stagger: 0.06, ease: "power2.out", delay: 0.15 }
    );
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("#predictBtn");
    const fd = new FormData(form);
    const payload = {
      income: +fd.get("income"),
      credit: +fd.get("credit"),
      annuity: +fd.get("annuity"),
      employed_years: +fd.get("employed_years"),
      age: +fd.get("age"),
      family_members: +fd.get("family_members"),
      education: fd.get("education"),
      contract_type: fd.get("contract_type"),
    };
    btn.disabled = true;
    status.textContent = "SCORING AGAINST LIVE MODEL…";
    status.classList.remove("is-error");
    try {
      const res = await fetch(`${window.API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`API responded ${res.status}`);
      const data = await res.json();
      status.textContent = `OK — ${data.risk_tier} TIER · MODEL API v1`;
      animateGauge(data.probability);
      renderWaterfall(data.shap_top_factors);
    } catch (err) {
      status.textContent = `API UNREACHABLE — IS THE SERVICE RUNNING AT ${window.API_BASE}? (${err.message})`;
      status.classList.add("is-error");
    } finally {
      btn.disabled = false;
    }
  });

  /* ------------------------------------------------ loader */
  const loader = $("#loader");
  const loaderCount = $("#loaderCount");
  const loaderBar = $("#loaderBar");

  function hideLoader() {
    gsap.to(loader, {
      autoAlpha: 0, duration: 0.45, ease: "power2.inOut",
      onComplete: () => {
        loader.style.display = "none";
        ScrollTrigger.refresh();
      },
    });
  }

  const warmup = fetch(`${window.API_BASE}/health`).then(() => true).catch(() => false);
  const minTime = new Promise((r) => setTimeout(r, REDUCED ? 200 : 1100));
  const fonts = document.fonts ? document.fonts.ready : Promise.resolve();

  if (REDUCED) {
    Promise.all([warmup, minTime]).then(() => { loader.style.display = "none"; ScrollTrigger.refresh(); });
  } else {
    const prog = { v: 0 };
    gsap.to(prog, {
      v: 92, duration: 1.2, ease: "power2.out",
      onUpdate: () => {
        loaderCount.textContent = `${Math.round(prog.v)}%`;
        loaderBar.style.width = `${prog.v}%`;
      },
    });
    Promise.all([warmup, minTime, fonts]).then(() => {
      gsap.to(prog, {
        v: 100, duration: 0.25,
        onUpdate: () => {
          loaderCount.textContent = `${Math.round(prog.v)}%`;
          loaderBar.style.width = `${prog.v}%`;
        },
        onComplete: hideLoader,
      });
    });
  }
})();
