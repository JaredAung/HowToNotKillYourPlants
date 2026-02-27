import type { PlantRec } from "@/app/components/PlantCard";

export const CHAT_SELECTED_PLANT_KEY = "chatSelectedPlant";
export const CHAT_RECOMMENDATIONS_KEY = "chatRecommendations";

export function setChatContext(plant?: PlantRec | null, recommendations?: PlantRec[] | null) {
  if (typeof window === "undefined") return;
  if (plant != null) {
    sessionStorage.setItem(CHAT_SELECTED_PLANT_KEY, JSON.stringify(plant));
  } else {
    sessionStorage.removeItem(CHAT_SELECTED_PLANT_KEY);
  }
  if (recommendations != null && recommendations.length > 0) {
    sessionStorage.setItem(CHAT_RECOMMENDATIONS_KEY, JSON.stringify(recommendations.slice(0, 5)));
  } else {
    sessionStorage.removeItem(CHAT_RECOMMENDATIONS_KEY);
  }
}

export function getChatContext(): {
  selectedPlant: PlantRec | null;
  recommendedPlants: PlantRec[];
} {
  if (typeof window === "undefined") {
    return { selectedPlant: null, recommendedPlants: [] };
  }
  let selectedPlant: PlantRec | null = null;
  try {
    const stored = sessionStorage.getItem(CHAT_SELECTED_PLANT_KEY);
    if (stored) selectedPlant = JSON.parse(stored) as PlantRec;
  } catch {
    sessionStorage.removeItem(CHAT_SELECTED_PLANT_KEY);
  }
  let recommendedPlants: PlantRec[] = [];
  try {
    const stored = sessionStorage.getItem(CHAT_RECOMMENDATIONS_KEY);
    if (stored) recommendedPlants = JSON.parse(stored) as PlantRec[];
  } catch {
    sessionStorage.removeItem(CHAT_RECOMMENDATIONS_KEY);
  }
  return { selectedPlant, recommendedPlants };
}

export function clearChatContext() {
  if (typeof window === "undefined") return;
  sessionStorage.removeItem(CHAT_SELECTED_PLANT_KEY);
  sessionStorage.removeItem(CHAT_RECOMMENDATIONS_KEY);
}
