import { ImageResponse } from "next/og";
import { BrandGlyph } from "@/components/brand";

export const runtime = "edge";
export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  // ImageResponse cannot resolve CSS variables; these are fixed dark-brand colors.
  return new ImageResponse(<div style={{ width: "100%", height: "100%", display: "flex",
    alignItems: "center", justifyContent: "center", background: "#1c2027", border: "1px solid #363c46",
    borderRadius: 8, color: "#cdb476" }}><BrandGlyph /></div>, size);
}
