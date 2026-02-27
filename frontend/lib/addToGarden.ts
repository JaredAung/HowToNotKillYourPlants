import type { PlantRec } from "@/app/components/PlantCard";

export const ADD_TO_GARDEN_PLANT_KEY = "addToGardenPlant";

export function navigateToAddToGarden(plant: PlantRec, router: { push: (path: string) => void }) {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(ADD_TO_GARDEN_PLANT_KEY, JSON.stringify(plant));
  router.push("/garden/add");
}
