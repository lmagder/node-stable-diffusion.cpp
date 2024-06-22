declare module "node-stable-diffusion.cpp" {
  export type Image = Readonly<{
    width: number;
    height: number;
    channel: number;
    data: Buffer;
  }>;

  export type Context = Readonly<{
    dispose: () => void;
    txt2img: (params: {
      prompt: string;
      negativePrompt?: string;
      clipSkip?: number;
      cfgScale?: number;
      width?: number;
      height?: number;
      sampleMethod?: number;
      sampleSteps?: number;
      seed?: number;
      batchCount?: number;
      controlCond?: Image;
      controlStrength?: number;
      styleRatio?: number;
      normalizeInput?: boolean;
      inputIdImagesPath?: string;
    }) => Promise<Image[]>;
  }>;

  export const createContext: (
    params: {
      model: string;
      vae?: string;
      taesd?: string;
      controlNet?: string;
      loraDir?: string;
      embedDir?: string;
      stackedIdEmbedDir?: string;
      vaeDecodeOnly?: boolean;
      vaeTiling?: boolean;
      freeParamsImmediately?: boolean;
      numThreads?: number;
      weightType?: number;
      cudaRng?: boolean;
      schedule?: number;
      keepClipOnCpu?: boolean;
      keepControlNetOnCpu?: boolean;
      keepVaeOnCpu?: boolean;
    },
    logCallback?: (level: "error" | "warn" | "info" | "debug", msg: string) => void,
    progressCallback?: (step: number, steps: number, time: number) => void
  ) => Promise<Context>;

  export const getSystemInfo: () => string;
  export const getNumPhysicalCores: () => number;
  export const weightTypeName: (weightType: number) => string;
}
