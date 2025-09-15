import React, { useState, useRef } from "react"
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib"

type Props = {
  args: {
    default?: string
    suggestion?: string
    input_color?: string
    suggestion_color?: string
    font_size?: string
    box_bg_color?: string
    height?: number
}
}

const MyComponent = ({ args }: Props) => {
  const [value, setValue] = useState(args.default || "")
  const suggestion = args.suggestion || ""
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const inputColor = args.input_color || "#0057b8"
  const suggestionColor = args.suggestion_color || "#b22222"
  const fontSize = args.font_size || "1.2em"

  const boxBgColor = args.box_bg_color || "#fff"
  const height = args.height || 300

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
    Streamlit.setComponentValue(e.target.value)
  }
  

  return (
    <>
    <style>
      {`
        html, body {
          background: ${boxBgColor} !important;
          width: 100%;
        }
      `}
    </style>

    <div
      style={{
        position: "relative",
        width: "100%",
        minHeight: `${height}px`,
        background: boxBgColor,
        boxSizing: "border-box",
        overflow: "hidden",
      }}
    >
    <textarea
      ref={textareaRef}
      value={value}
      onChange={handleChange}
      rows={10}
      style={{
        width: "100%",
        height: `${height}px`,
        fontSize: fontSize,
        position: "relative",
        zIndex: 2,
        background: "transparent",
        color: inputColor,
        border: "1px solid #000000",
        resize: "none",
        boxSizing: "border-box",
        borderRadius: "8px",
      }}
    />
    {/* Overlay suggestion at end of text */}
    <div
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        background: boxBgColor,
        color: "transparent",
        opacity: 1,
        pointerEvents: "none",
        fontSize: fontSize,
        zIndex: 1,
        width: "100%",
        minHeight: `${height}px`,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}
    >
      {value}
      <span style={{ color: suggestionColor }}>{suggestion}</span>
    </div>
  </div>
  </>
)
}

export default withStreamlitConnection(MyComponent)