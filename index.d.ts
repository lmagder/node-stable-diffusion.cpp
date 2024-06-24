declare module "node-stable-diffusion-cpp" {
  declare enum SampleMethod {
    EulerA,
    Euler,
    Heun,
    DPM2,
    DPMPP2SA,
    DPMPP2M,
    DPMPP2Mv2,
    LCM,
  }

  declare enum Schedule {
    Default,
    Discrete,
    Karras,
    AYS,
  }

  declare enum Type {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
  }

  export type Image = Readonly<{
    width: number;
    height: number;
    channel: 3 | 4;
    data: Buffer;
  }>;

  export type Context = Readonly<{
    dispose: () => Promise<void>;
    txt2img: (params: {
      prompt: string;
      negativePrompt?: string;
      clipSkip?: number;
      cfgScale?: number;
      width?: number;
      height?: number;
      sampleMethod?: SampleMethod;
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
      weightType?: Type;
      cudaRng?: boolean;
      schedule?: Schedule;
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
