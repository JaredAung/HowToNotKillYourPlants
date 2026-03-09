import type { PlantRec } from "@/app/components/PlantCard";
import { addToGarden } from "@/lib/api";

export const ADD_TO_GARDEN_PLANT_KEY = "addToGardenPlant";

export function navigateToAddToGarden(plant: PlantRec, router: { push: (path: string) => void }) {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(ADD_TO_GARDEN_PLANT_KEY, JSON.stringify(plant));
  router.push("/garden/add");
}

/** Add plant directly to garden via API. Does not navigate. Caller shows success UI. */
export async function addDirectlyToGarden(plant: PlantRec): Promise<string> {
  await addToGarden(plant.plant_id, undefined);
  return plant.common_name ?? plant.latin ?? `Plant #${plant.plant_id}`;
}
