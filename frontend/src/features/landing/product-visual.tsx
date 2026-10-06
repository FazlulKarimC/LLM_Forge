import Image from "next/image";
import { productPreviews, type PreviewId } from "./product-previews";

export function ProductVisual({ id, priority = false }: { id: PreviewId; priority?: boolean }) {
  const preview = productPreviews.find((item) => item.id === id)!;
  return <div className="product-visual">
    {(["dark", "light"] as const).map((theme) => <picture key={theme} className={`preview-${theme}`}>
      <source media="(max-width: 639px)" srcSet={`/walkthrough/${id}-${theme}-mobile.jpg`} width={preview.mobileWidth} height={preview.mobileHeight} />
      <Image src={`/walkthrough/${id}-${theme}.jpg`} alt={preview.alt} width={1153} height={783}
        sizes="(max-width: 639px) 100vw, (max-width: 1200px) 90vw, 1120px" priority={priority} />
    </picture>)}
  </div>;
}
