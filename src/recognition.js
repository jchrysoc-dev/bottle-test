import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from '@huggingface/transformers';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
let processor, model;
export async function loadRecognitionModel(onProgress = () => {}) {
  if (model) return;
  onProgress('Loading bottle recognition model…');
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  try { model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8' }); }
  catch { model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID); }
  onProgress('Model ready.');
}
export async function embedBottle(file) {
  const image = await RawImage.fromBlob(file);
  const inputs = await processor(image);
  const { image_embeds } = await model(inputs);
  return Array.from(image_embeds.normalize().data);
}
export const cosine = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
export const modelId = MODEL_ID;
