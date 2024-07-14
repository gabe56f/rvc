<script>
    import ScrollArea from "../scroll-area/scroll-area.svelte";

    export let open = false;
</script>

<aside
	class="z-30 relative h-full max-w-lg w-full rounded-l shadow-lg border-r border-gray-100 bg-opacity-25 backdrop-blur-lg bg-slate-500"
	class:open
>
	<div class="hidden max-sm:block min-w-full">
		<slot name="mobile" />
	</div>
	<ScrollArea class="h-full">
		<slot />
	</ScrollArea>
</aside>

<style>
	aside {
		left: -100%;
		transition: left 0.2s ease-in-out;
	}

	.open {
		left: 0;
	}
</style>