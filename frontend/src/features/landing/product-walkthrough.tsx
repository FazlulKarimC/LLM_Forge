"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { productPreviews } from "./product-previews";
import { ProductVisual } from "./product-visual";

export function ProductWalkthrough() {
  const [selected, setSelected] = useState(0);
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number;
    if (event.key === "ArrowRight") next = (index + 1) % productPreviews.length;
    else if (event.key === "ArrowLeft") next = (index + productPreviews.length - 1) % productPreviews.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = productPreviews.length - 1;
    else return;
    event.preventDefault(); setSelected(next); tabs.current[next]?.focus();
  }
  return <section id="walkthrough" aria-labelledby="walkthrough-title" className="landing-walkthrough">
    <div className="landing-section-intro">
      <div><p className="section-label">A look inside</p><h2 id="walkthrough-title">One workspace. Three ways to improve.</h2></div>
      <p>Explore real screens. No sign-in needed for this walkthrough.</p>
    </div>
    <div className="walkthrough-frame">
      <div className="walkthrough-toolbar">
        <div role="tablist" aria-label="Product walkthrough">
          {productPreviews.map((preview, index) => <button key={preview.id} type="button" role="tab"
            id={`preview-tab-${preview.id}`} aria-controls={`preview-panel-${preview.id}`}
            aria-selected={selected === index} tabIndex={selected === index ? 0 : -1}
            ref={(element) => { tabs.current[index] = element; }}
            onKeyDown={(event) => navigate(event, index)} onClick={() => setSelected(index)}>{preview.label}</button>)}
        </div>
        <span className="walkthrough-badge"><span />Read-only examples</span>
      </div>
      {productPreviews.map((preview, index) => <div key={preview.id} role="tabpanel"
        id={`preview-panel-${preview.id}`} aria-labelledby={`preview-tab-${preview.id}`}
        tabIndex={0} hidden={selected !== index}>
        <ProductVisual id={preview.id} priority={index === 0} />
        <div className="walkthrough-caption"><div><h3>{preview.title}</h3><p>{preview.description}</p></div>
          <p className="walkthrough-disclosure">{preview.caption}</p></div>
      </div>)}
    </div>
  </section>;
}
