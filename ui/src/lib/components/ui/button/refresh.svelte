<script>
    import { RotateCw } from "lucide-svelte";
    import Button from "./button.svelte";

    export let refresh;
</script>

<Button
    on:click={refresh}
    class="fixed right-16 bottom-3 z-50"
    variant="outline"
    size="icon"
>
    <RotateCw class="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-100" />
    <span class="sr-only">Toggle theme</span>
</Button>
