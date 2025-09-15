import React, { useState } from "react"
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib"

const MyComponent = () => {
  const [value, setValue] = useState("")

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue(e.target.value)
    Streamlit.setComponentValue(e.target.value) // Send value to Python on every change
  }

  return (
    <div>
      <input
        type="text"
        id="inputBox"
        placeholder="Type something..."
        value={value}
        onChange={handleChange}
        style={{ width: "300px", fontSize: "1.2em" }}
      />
      <p>Output: <span id="outputLabel">{value}</span></p>
    </div>
  )
}

export default withStreamlitConnection(MyComponent)