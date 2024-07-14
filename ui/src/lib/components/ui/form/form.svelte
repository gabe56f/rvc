<script lang="ts">
import * as HoverCard from "../hover-card";
import Label from "../label/label.svelte";

    export let cclass: string = "w-full";
    export let label: string;
</script>

<HoverCard.Root>
    <!--aaa-->
    <HoverCard.Trigger>
        <div class={cclass + " p-4 block"}>
            <Label>{label}</Label>
            <p />
            <slot name="main" />
        </div>
    </HoverCard.Trigger>
    {#if $$slots.hover}
        <HoverCard.Content>
            <slot name="hover" />
        </HoverCard.Content>
    {/if}
</HoverCard.Root>