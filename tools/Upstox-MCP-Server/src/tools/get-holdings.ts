import { z } from "zod";
import { ToolHandler, ToolResponse, GetHoldingsArgs, Env } from "../types";
import { 
  UPSTOX_API_BASE_URL, 
  UPSTOX_API_HOLDINGS_ENDPOINT,
  HEADERS,
  ERROR_MESSAGES
} from "../constants";
import { UPSTOX_CONFIG } from "../config";

export const getHoldingsSchema = {
  // accessToken: z.string().min(1, "Access token is required"),
};

// const GetHoldingsArgsSchema = z.object(getHoldingsSchema);

interface Holding {
  isin: string;
  cnc_used_quantity: number;
  collateral_type: string;
  company_name: string;
  haircut: number;
  product: string;
  quantity: number;
  trading_symbol: string;
  tradingsymbol: string;
  last_price: number;
  close_price: number;
  pnl: number;
  day_change: number;
  day_change_percentage: number;
  instrument_token: string;
  average_price: number;
  collateral_quantity: number;
  collateral_update_quantity: number;
  t1_quantity: number;
  exchange: string;
}

interface UpstoxHoldingsResponse {
  status: string;
  data: Holding[];
}

export const getHoldingsHandler: ToolHandler<GetHoldingsArgs, Env> = async (args: GetHoldingsArgs, env: Env): Promise<ToolResponse> => {
  // Use token from config if not provided in args
  // const accessToken = args.accessToken || UPSTOX_CONFIG.ACCESS_TOKEN;

  // const accessToken = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiJBSzk0NjEiLCJqdGkiOiI2ODY2ZWUzY2M5NGRmNjRkMzVhOTkxZWUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc1MTU3NjEyNCwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzUxNTgwMDAwfQ.qDsKoZKPPR3Zg9_Xt38nnSpnpA-PAx6bYxx5pEONbl0";
  // const validatedArgs = GetHoldingsArgsSchema.parse({ accessToken });
  try {
  const response = await fetch(`${UPSTOX_API_BASE_URL}${UPSTOX_API_HOLDINGS_ENDPOINT}`, {
    method: "GET",
    headers: {
      "Accept": HEADERS.ACCEPT,
      "Authorization": `Bearer ${env.UPSTOX_ACCESS_TOKEN}`
    }
  });

  if (!response.ok) {
    // console.log(await response.text());
    throw new Error(ERROR_MESSAGES.API_ERROR);
  }

  const data = await response.json() as UpstoxHoldingsResponse;
  
  return {
    content: [{
      type: "text",
      text: JSON.stringify(data, null, 2)
    
    }]
  
  };
 }
 catch (error) {
  console.error("Error fetching holdings:", error);
  return {
    content: [{
      type: "text",
      text: ERROR_MESSAGES.API_ERROR
    }]
  };
 }
};