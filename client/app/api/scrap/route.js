// app/api/scrap/route.js

import { NextResponse } from "next/server";

export async function POST(request) {
  const { stock, platforms, days, max_tweets } = await request.json();

  // const apiUrl = `http://localhost:8000/scrap`;
  const apiUrl = `https://stocks-sentiment-analysis.fly.dev/scrap`;

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ stock, platforms, days, max_tweets }),
    });

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data); // Return the data to the client
    } else {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || "Failed to fetch data" },
        { status: response.status }
      );
    }
  } catch (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
