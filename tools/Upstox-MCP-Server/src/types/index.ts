import { z } from "zod";

export interface Env {
  UPSTOX_ACCESS_TOKEN: string;
}

export interface ToolResponse {
  [key: string]: unknown;
  content: Array<{
    type: "text";
    text: string;
  } | {
    type: "image";
    data: string;
    mimeType: string;
  } | {
    type: "resource";
    resource: {
      text: string;
      uri: string;
      mimeType?: string;
    } | {
      uri: string;
      blob: string;
      mimeType?: string;
    };
  }>;
  _meta?: {
    [key: string]: unknown;
  };
  isError?: boolean;
}

export interface ToolHandler<T, E={ [key: string]: unknown }> {
  (args: T, extra: E): Promise<ToolResponse>;
}

export interface GetProfileArgs {
  accessToken: string;
}

export interface GetFundsMarginArgs {
  accessToken: string;
  segment?: 'SEC' | 'COM';
}

export interface GetHoldingsArgs {
  // accessToken: string;
}

export interface GetPositionsArgs {
  accessToken: string;
}

export interface GetMtfPositionsArgs {
  accessToken: string;
}

export interface GetOrderDetailsArgs {
  accessToken: string;
  orderId: string;
} 